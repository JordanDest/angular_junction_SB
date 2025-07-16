#data_manager.py
from __future__ import annotations
import csv
import logging
import os
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
# Internal deps (must already exist in your project)
try:
    from .kraken_api import KrakenREST
except ImportError:
    from kraken_api import KrakenREST
from utils.file_store import make_path, atomic_write
try:
    from .file_store import make_path, atomic_write
except ImportError:
    from file_store import make_path, atomic_write
   # must expose .public()
def safe_convert(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

__all__ = [
    "DataValidationError",
    "DataFetchError",
    "CoinCfg",
    "DataManager",
]

###############################################################################
# Globals & type aliases
###############################################################################
OHLC_COLUMNS: list[str] = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "count",
]
CsvRow = List[str | float]
CoinCfg = Dict[str, Any]

_DEFAULT_DAYS_BACK = 55
_DEFAULT_INTERVAL_MINUTES = 1

# Kraken returns paging tokens in **nanoseconds** since Unix epoch.
_NANO = 1_000_000_000

_LOG = logging.getLogger(__name__)

###############################################################################
# Exceptions
###############################################################################


class DataValidationError(RuntimeError):
    """Raised when OHLC data fail the *strict* validation rules."""


class DataFetchError(RuntimeError):
    """Raised when Kraken returns an error or the response schema is unexpected."""

###############################################################################
# Helpers – trade aggregation & validation
###############################################################################


def _aggregate_trades(
    trades: Iterable[list[Any]],
    interval_seconds: int,
) -> list[CsvRow]:
    """Convert raw Kraken trades → OHLC rows (1 row = 1 candle).

    Kraken trade row = ``[price, volume, time, side, orderType, misc, id?]``.
    ``time`` is seconds with millisecond decimals.
    """
    buckets: dict[int, Dict[str, Any]] = {}
    for p, v, ts, *_ in trades:
        try:
            price = float(p)
            vol = float(v)
            sec = int(float(ts))  # truncate ms
        except Exception:
            continue

        bucket = sec - (sec % interval_seconds)
        b = buckets.get(bucket)
        if b is None:
            buckets[bucket] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "vol": vol,
                "pv": price * vol,
                "cnt": 1,
            }
        else:
            b["high"] = max(b["high"], price)
            b["low"] = min(b["low"], price)
            b["close"] = price
            b["vol"] += vol
            b["pv"] += price * vol
            b["cnt"] += 1

    rows: list[CsvRow] = []
    for bucket_ts in sorted(buckets):
        b = buckets[bucket_ts]
        iso = datetime.fromtimestamp(bucket_ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        rows.append(
            [
                iso,
                b["open"],
                b["high"],
                b["low"],
                b["close"],
                b["pv"] / b["vol"],
                b["vol"],
                b["cnt"],
            ]
        )
    return rows


###############################################################################
# Validation helpers – stateless, tolerant
###############################################################################


# Timestamp cleanup helper
def _sanitize_timestamps(series: pd.Series) -> pd.Series:
    """
    Normalise every element of *series* to valid ISO‑8601:

    • space → 'T'
    • collapse any '::' (repeat until gone)
    • ensure seconds field present (HH:MM  →  HH:MM:00)
    • Z → +00:00
    • +00:000 / +00::00 → +00:00
    """
    s = series.astype(str).str.strip()

    # 1. date separator
    s = s.str.split().str.join("T")

    # 2. nuke every run of ≥2 colons (do twice to be safe after later inserts)
    for _ in range(2):
        s = s.str.replace(r":{2,}", ":", regex=True)

    # 3. if only HH:MM(±…|Z|$), append ':00'
    s = s.str.replace(r"(T\d{2}:\d{2})(?=[+\-Z]|$)", r"\1:00", regex=True)

    # 4. canonicalise offsets
    s = (
        s.str.replace(r"Z$", "+00:00", regex=True)
         .str.replace(r"\+00:000$", "+00:00", regex=True)
         .str.replace(r"\+00::00$", "+00:00", regex=True)
    )

    # 5. final double‑colon sweep (after we may have inserted)
    s = s.str.replace(r":{2,}", ":", regex=True)

    # 6. drop trailing “+00:00” that the old files still contain
   # s = s.str.replace(r'\+00:00$', '', regex=True)
   # keep any trailing “+00:00” or “Z” intact so pandas can see the offset?

    return s


def _validate_dataframe_ohlc(
    df: pd.DataFrame,
    *,
    interval_seconds: int,
    expected_rows: int | None = None,
    check_freshness: bool = False,        # ← NEW
) -> None:
    """Raise *DataValidationError* if *df* breaks any rule."""

    # 1. strict column ordering
    if list(df.columns) != OHLC_COLUMNS:
        raise DataValidationError("Unexpected CSV column ordering")

     # 2. timestamp sanitation + parsing
    # raw = _sanitize_timestamps(df["time"])
    # ts = pd.to_datetime(raw,  utc=True, format='ISO8601', errors='coerce')
    raw = _sanitize_timestamps(df["time"])
    # attempt to let pandas infer the exact ISO‐8601 variant (space/T, offsets, sub‐seconds…)
    ts  = pd.to_datetime(raw, utc=True, errors='coerce')
    # 2b. numeric epoch fallback
    if ts.isna().any():
        mask = ts.isna()
        nums = pd.to_numeric(raw[mask], errors="coerce")
        if nums.notna().any():
            sec = nums.lt(10**11)
            ts.loc[mask & sec]  = pd.to_datetime(nums[sec].astype("int64"), unit="s", utc=True)
            ts.loc[mask & ~sec] = pd.to_datetime((nums[~sec] // 1000).astype("int64"), unit="s", utc=True)

    # 2c. final cleanup — if any still remain NaT, drop & warn
    if ts.isna().any():
        bad_rows = ts.isna().sum()
        samples  = list(raw[ts.isna()].unique()[:5])
        _LOG.warning("‼️  dropping %s unparseable rows, e.g. %s", bad_rows, samples)
        df = df.loc[~ts.isna()].copy()
        ts = ts.loc[~ts.isna()]

    df["time"] = ts.sort_values().values  # enforce chronological order

    # 3. gap detection
    #print(df["time"].diff().dt.total_seconds().describe())

    if (df["time"].diff().dt.total_seconds().fillna(interval_seconds) > interval_seconds).any():
        raise DataValidationError("Gaps larger than interval detected")

    # 4. OHLC sanity
    bad = (
        (df["high"]  < df["low"])  |
        (df["high"]  < df["open"]) |
        (df["high"]  < df["close"])|
        (df["low"]   > df["open"]) |
        (df["low"]   > df["close"])|
        (df["volume"] < 0)         |
        (df["count"]  < 0)
    )
    if bad.any():
        raise DataValidationError(f"{bad.sum()} invalid OHLC rows detected")

    # 5. size & freshness -----------------------------------------------
    if expected_rows is not None and len(df) < expected_rows:
        raise DataValidationError(f"Too few rows: {len(df)} < {expected_rows}")

    if check_freshness:                         # ← NEW guard
        now_bucket = pd.Timestamp.utcnow().floor(f"{interval_seconds}s")
        if now_bucket.tz is None:
            now_bucket = now_bucket.tz_localize("UTC")
        lag_sec = (now_bucket - df["time"].iloc[-1]).total_seconds()
        if lag_sec > 3 * interval_seconds:      # allow **3 buckets** behind
            raise DataValidationError(
                f"Dataset stale – lag {int(lag_sec)} s (> {3*interval_seconds}s)"
            )

    
###############################################################################
# Gap repair helpers
###############################################################################
def _fill_missing_buckets(df: pd.DataFrame, *, interval_seconds: int) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    # Sanitize + parse with robust ISO parser
    raw = _sanitize_timestamps(df["time"])
    #ts  = pd.to_datetime(raw, utc=True, format='ISO8601', errors='coerce')
    ts  = pd.to_datetime(raw, utc=True, errors='coerce')
    if ts.isna().any():
        bad_rows = ts.isna().sum()
        samples = list(raw[ts.isna()].unique()[:5])
        _LOG.warning("‼️  dropping %s unparseable rows, e.g. %s", bad_rows, samples)
        df = df.loc[~ts.isna()].copy()
        ts = ts.loc[~ts.isna()]

    df["time"] = ts
    df = df.sort_values("time")  # ensure ordering after parsing
    df = df.drop_duplicates(subset="time", keep="last")

    start = df["time"].iloc[0].floor(f"{interval_seconds}s")
    end   = df["time"].iloc[-1].ceil(f"{interval_seconds}s")

    full_idx = pd.date_range(start, end, freq=f"{interval_seconds}s", tz="UTC")

    before = len(df)
    df = df.drop_duplicates(subset="time", keep="last")

    df = (
        df.set_index("time")
          .reindex(full_idx)
          .sort_index()
    )

    price_cols = ["open", "high", "low", "close", "vwap"]
    df[price_cols] = df[price_cols].ffill()
    df[["volume", "count"]] = df[["volume", "count"]].fillna(0)

    filled = len(df) - before
    df = df.reset_index(names="time")
    df["time"] = (
        df["time"]
          .dt.tz_localize(None)                 # drop UTC offset safely
          .dt.strftime("%Y-%m-%dT%H:%M:%S")     # always 'YYYY‑MM‑DDTHH:MM:SS'
    )
    gaps = pd.to_datetime(df["time"]).diff().dt.total_seconds()
    
    large_gaps = gaps[gaps > (60*7)]  # gaps longer than 7 min
   # print("testing:    ", large_gaps)
    return df, filled





###############################################################################
# Kraken accessor using /Trades
###############################################################################


@dataclass(slots=True)
class _KrakenAccessor:
    client: KrakenREST
    RATE_LIMIT_SEC: float = 1.0

    # ------------------------------------------------------------------ #
    # low‑level wrapper
    # ------------------------------------------------------------------ #
    def _public(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        resp = self.client.public(method, params)

        if not isinstance(resp, dict):
            raise DataFetchError("Malformed Kraken response – not a JSON dict")

        if resp.get("error"):
            raise DataFetchError("Kraken API error: " + ", ".join(resp["error"]))

        # Accept both: ➊ raw (contains "result") ➋ already‑unwrapped (no "result")
        result = resp.get("result", resp)

        if not isinstance(result, dict) or "last" not in result:
            _LOG.warning("‼️  Raw Kraken response for %s: %s", method, resp)
            raise DataFetchError("Unexpected Kraken schema – cannot locate trade book")

        _LOG.debug("Kraken %s %.3f s", method, time.perf_counter() - t0)
        return result

    # ------------------------------------------------------------------ #
    # high‑level helpers
    # ------------------------------------------------------------------ #
    def trades(
        self, *, pair: str, since: int | None = None
    ) -> tuple[list[list[Any]], int]:
        """Return *trades, last_token* exactly as Kraken calls for paging."""
        params: Dict[str, Any] = {"pair": pair}
        if since is not None:
            params["since"] = since
        result = self._public("Trades", params)

        # result is {<pair>: [...trades...], "last": <token>}
        pair_key = next(k for k in result if k != "last")
        return result[pair_key], int(result["last"])

    def paged_trades(
        self, *, pair: str, start_token: int = 0
    ) -> Iterable[list[list[Any]]]:
        """Yield successive trade pages starting from *start_token*.

        *start_token* == 0 means “oldest Kraken keeps in cache”.
        """
        since = start_token
        while True:
            page, since_next = self.trades(pair=pair, since=since)
            if not page:
                return
            yield page
            if since_next == since:  # nothing new
                return
            since = since_next
            time.sleep(self.RATE_LIMIT_SEC)


###############################################################################
# DataManager – public facade
###############################################################################


class DataManager:
    """Owns CSV OHLC datasets for a list of Kraken symbols (trades edition)."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        coin_settings: Sequence[CoinCfg],
        *,
        data_directory: str | Path = "data",
        default_interval_minutes: int = _DEFAULT_INTERVAL_MINUTES,
        default_days_back: int = _DEFAULT_DAYS_BACK,
        verbose: bool = False,
        interactive: bool | None = None,
    ) -> None:
        self.coin_settings: list[CoinCfg] = list(coin_settings)
        self.data_path = Path(data_directory)
        self.interval_minutes_default = default_interval_minutes
        self.days_back_default = default_days_back
        self.verbose = verbose
        self.interactive = sys.stdin.isatty() if interactive is None else interactive

        self.data_path.mkdir(parents=True, exist_ok=True)

        self._kraken = _KrakenAccessor(KrakenREST())
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._stop_event = threading.Event()
        self._validator = _validate_dataframe_ohlc

        self._stream_threads: list[threading.Thread] = []
        self._refresher_thread: threading.Thread | None = None

        _LOG.info(
            "DataManager initialised – directory=%s symbols=%s",
            self.data_path.resolve(),
            [c["symbol"] for c in self.coin_settings],
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ensure_datasets(self) -> None:
        for cfg in self.coin_settings:
            self._ensure_dataset(cfg)

    def start(
        self,
        *,
        refresh_days: int = 30,
        refresh_interval_seconds: int = 3_600,
    ) -> None:
        self.ensure_datasets()

        # per‑symbol streamers
        for cfg in self.coin_settings:
            t = threading.Thread(
                target=self._stream_symbol_forever,
                args=(cfg,),
                daemon=True,
                name=f"stream-{cfg['symbol']}",
            )
            t.start()
            self._stream_threads.append(t)

        # global refresher
        self._refresher_thread = threading.Thread(
            target=self._refresher_loop,
            args=(refresh_days, refresh_interval_seconds),
            daemon=True,
            name="refresher",
        )
        self._refresher_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._stream_threads:
            t.join(timeout=5)
        if self._refresher_thread is not None:
            self._refresher_thread.join(timeout=5)
        _LOG.info("All DataManager threads stopped")



    ###############################################################################
    # Dataset helpers – newest file & age check
    ###############################################################################
    _MAX_DATA_AGE_DAYS = 10         # <‑‑ configurable threshold

    def _latest_csv(self, symbol: str) -> Path | None:
        """Return newest CSV for *symbol* or *None* if nothing on disk."""
        files = sorted(
            self.data_path.glob(f"{symbol}_*.csv"),
            key=os.path.getmtime,
            reverse=True,
        )
        return files[0] if files else None
    # ------------------------------------------------------------------ #
    # Public helper – lets external tools locate the freshest dataset
    # ------------------------------------------------------------------ #
    def latest_csv(self, symbol: str) -> Path | None:
        """
        Public alias for :meth:`_latest_csv`.
        Added so utility modules (quick_score, dashboards, etc.) don’t have
        to reach into a “private” member or re‑implement the glob logic.
        """
        return self._latest_csv(symbol)
    
    def get_csv(
        self,
        symbol: str,
        *,
        interval_minutes: int | None = None,
        bar: int | None = None,
        days_back: int | None = None,
        backfill: bool = True,
    ) -> Path:
        """
        Ensure the requested CSV exists locally; back‑fill if necessary,
        then return the path.
        """
        days_back = days_back or self.days_back_default
        path = self._csv_path(
            symbol,
            interval_minutes=interval_minutes,
            days_back=days_back,
        )
        if backfill and not path.exists():
            cfg = {"symbol": symbol,
                   "interval_minutes": interval_minutes,
                   "days_back": days_back}
            self._perform_backfill(cfg)
        return path
    #######################################################################
    def _is_stale(self, path: Path) -> bool:
        """True if *path* is older than the freshness threshold."""
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.utcnow().replace(tzinfo=timezone.utc) - mtime).days >= self._MAX_DATA_AGE_DAYS

    # ------------------------------------------------------------------ #
    # Internal – dataset bootstrap
    # ------------------------------------------------------------------ #
    def _ensure_dataset(self, cfg: CoinCfg) -> None:
        sym = cfg["symbol"]
        int_min = (
            cfg.get("interval_minutes")
            if cfg.get("interval_minutes") is not None
            else cfg.get("bar", self.interval_minutes_default)
        )
        int_sec = int_min * 60


        # pick newest file (if any)
        csv_candidates = sorted(self.data_path.glob(f"{sym}_*.csv"), key=os.path.getmtime, reverse=True)
        curr_path: Path | None = csv_candidates[0] if csv_candidates else None
        if curr_path is None:
            _LOG.warning("%s: dataset missing – running back‑fill", sym)
            self._perform_backfill(cfg)
            return
#         # ───────────────────── age gate ───────────────────── #
#         if self._is_stale(curr_path):
#             _LOG.warning("%s: dataset %s is >%sd old – running back‑fill",
#                          sym, curr_path.name, _MAX_DATA_AGE_DAYS)
#             self._perform_backfill(cfg)
#             return  
        try:
            df = pd.read_csv(curr_path)
            # run the sanitizer once in case this is an OLD file
            df["time"] = _sanitize_timestamps(df["time"])
            df, _ = _fill_missing_buckets(df, interval_seconds=int_sec)
            self._validator(df, interval_seconds=int_sec, check_freshness=False)
            # if it passes, rewrite the clean file so we never hit this again
            self._atomic_write(df, curr_path)

        except Exception as exc:   # noqa: BLE001
            _LOG.warning("%s: dataset %s invalid (%s)", sym, curr_path.name, exc)
            # quarantine bad file and back‑fill afresh
            bad = curr_path.with_stem(curr_path.stem + "_failed")
            curr_path.rename(bad)
            _LOG.warning("%s: moved corrupt CSV to %s", sym, bad.name)
            self._perform_backfill(cfg)

    # ------------------------------------------------------------------ #
    # Internal – back‑fill (trades edition)
    # ------------------------------------------------------------------ #
    def _perform_backfill(self, cfg: CoinCfg) -> None:
        sym        = cfg["symbol"]
        int_min = (
            cfg.get("interval_minutes")
            if cfg.get("interval_minutes") is not None
            else cfg.get("bar", self.interval_minutes_default)
        )

        int_sec    = int_min * 60
        days_back  = cfg.get("days_back", self.days_back_default)
        target_rows = days_back * 24 * 60 // int_min

        # ❶ Build Kraken token
        start_since_ns = int((time.time() - days_back * 86_400) * _NANO)
        _LOG.info("%s: back‑filling %s days (%sm bars) via /Trades", sym, days_back, int_min)

        # ❷ Collect pages
        all_trades, buckets_seen = [], set[int]()
        pbar = tqdm(total=target_rows, desc=f"Backfill {sym}", unit="bars",
                    leave=False) if self.verbose and sys.stderr.isatty() else None
        try:
            for page in self._kraken.paged_trades(pair=sym, start_token=start_since_ns):
                all_trades.extend(page)
                for _p, _v, _ts, *_ in page:
                    try:
                        sec = int(float(_ts)); buckets_seen.add(sec - sec % int_sec)
                    except Exception:
                        pass
                if pbar:
                    pbar.update(len(buckets_seen) - pbar.n)
                    pbar.set_postfix(trades=len(all_trades))
        finally:
            if pbar:
                pbar.close()

        # ❸ Aggregate + gap‑fill
        df = pd.DataFrame(_aggregate_trades(all_trades, int_sec), columns=OHLC_COLUMNS)
        df, n_filled = _fill_missing_buckets(df, interval_seconds=int_sec)
        if n_filled:
            _LOG.warning("%s: filled %s empty buckets", sym, n_filled)

        # ❹ Validate and persist
        try:
            self._validator(df, interval_seconds=int_sec, expected_rows=target_rows)
            dest = self._csv_path(sym, interval_minutes=int_min, days_back=days_back)
            self._atomic_write(df, dest)
            _LOG.info("%s: back‑fill complete – %s rows → %s", sym, len(df), dest.name)
        except DataValidationError as exc:
            _LOG.error("%s: validation FAILED (%s) – saving as *_failed", sym, exc)
            fail = self._csv_path_failed(sym, interval_minutes=int_min, days_back=days_back)
            self._atomic_write(df, fail)


    # ------------------------------------------------------------------ #
    # Internal – refresher
    # ------------------------------------------------------------------ #
    def _refresher_loop(self, refresh_days: int, interval_seconds: int) -> None:
        while not self._stop_event.wait(interval_seconds):
            for cfg in self.coin_settings:
                self._refresh_symbol(cfg, window_days=refresh_days)

    def _refresh_symbol(self, cfg: CoinCfg, *, window_days: int) -> None:
        sym = cfg["symbol"]
        lock = self._locks[sym]
        int_min = (
            cfg.get("interval_minutes")
            if cfg.get("interval_minutes") is not None
            else cfg.get("bar", self.interval_minutes_default)
        )

        int_sec = int_min * 60
        path = self._latest_csv(sym) or self._csv_path(sym)

        with lock:
            try:
                df = pd.read_csv(path)
                self._validator(df, interval_seconds=int_sec)
                last_ts = int(
                    pd.to_datetime(df["time"], utc=True).iloc[-1].timestamp()
                )
            except Exception as exc:
                _LOG.warning("%s: refresher invalid dataset (%s)", sym, exc)
                return

            try:
                trades, _ = self._kraken.trades(
                    pair=sym, since=int(last_ts * _NANO)
                )
            except DataFetchError as exc:
                _LOG.warning("%s: refresher Kraken error (%s)", sym, exc)
                return

            bars = _aggregate_trades(trades, int_sec)
            if not bars:
                return

            df_new = pd.DataFrame(bars, columns=OHLC_COLUMNS)
            df_combined = pd.concat([df, df_new], ignore_index=True)

            # ---------------------------------------------------------- #
            # Deduplicate overlapping rows and roll a time‑window view.
            # ---------------------------------------------------------- #
            df_combined = (
                df_combined.drop_duplicates(subset="time", keep="last")
                .sort_values("time")
                .reset_index(drop=True)
            )

            cutoff_ts = time.time() - window_days * 86_400
            epoch_s = (
                pd.to_datetime(df_combined["time"], utc=True).view("int64")
                // _NANO
            )
            df_combined = df_combined[epoch_s >= cutoff_ts]
            expected_rows = window_days * 24 * 60 // int_min

            try:
                self._validator(
                    df_combined,
                    interval_seconds=int_sec,
                    expected_rows=expected_rows,
                )
            except DataValidationError as exc:
                _LOG.warning(
                    "%s: refreshed data invalid (%s) – aborting replace", sym, exc
                )
                return

            self._atomic_write(df_combined, path)
            _LOG.info(
                "%s: refresh +%s rows (total=%s)",
                sym,
                len(df_new),
                len(df_combined),
            )

    # ------------------------------------------------------------------ #
    # Internal – streaming (latest closed candle only)
    # ------------------------------------------------------------------ #
    def _stream_symbol_forever(self, cfg: CoinCfg) -> None:
        sym = cfg["symbol"]
        lock = self._locks[sym]
        int_min = (
            cfg.get("interval_minutes")
            if cfg.get("interval_minutes") is not None
            else cfg.get("bar", self.interval_minutes_default)
        )

        int_sec = int_min * 60
        path = self._latest_csv(sym) or self._csv_path(sym)
        last_written_ts: int | None = None

        while not self._stop_event.is_set():
            try:
                trades, _ = self._kraken.trades(pair=sym)
            except DataFetchError as exc:
                _LOG.warning("%s: stream Kraken error (%s)", sym, exc)
                time.sleep(int_sec)
                continue

            bars = _aggregate_trades(trades, int_sec)
            if not bars:
                time.sleep(int_sec)
                continue

            latest = bars[-1]
            ts_sec = int(
                datetime.fromisoformat(latest[0])
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
            if last_written_ts is not None and ts_sec <= last_written_ts:
                time.sleep(int_sec)
                continue

            with lock:
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerow(latest)
            last_written_ts = ts_sec
            time.sleep(int_sec)

    # ------------------------------------------------------------------ #
    # Internal – helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prompt_yes_no(
        prompt: str, *, timeout_seconds: int = 30, default: bool = False
    ) -> bool:
        if not sys.stdin.isatty():
            return default
        try:
            return input(prompt).strip().lower().startswith("y")
        except Exception:
            return default


    # ------------------------------------------------------------------ #

    def _csv_path(
        self,
        symbol: str,
        *,                                 # kwargs‑only from here on
        interval_minutes: int | None = None,
        bar: int | None = None,            # ← keep old spelling temporarily
        days_back: int | None = None,
    ) -> Path:
        """
        Return the canonical CSV path for (*symbol*, *interval_minutes*, *days_back*).

        New preferred kwarg ……  interval_minutes
        Deprecated (until 2025‑12‑31) ………  bar
        """
        if interval_minutes is None and bar is None:
            raise TypeError("Need interval_minutes=<int>")

        if interval_minutes is None:       # caller used legacy name
            warnings.warn(
                "`bar=` is deprecated; use interval_minutes=",
                DeprecationWarning,
                stacklevel=2,
            )
            interval_minutes = bar

        days_back = (
            days_back if days_back is not None else self.days_back_default
        )
        return make_path(
            self.data_path,
            symbol,
            days_back=days_back,
            interval_minutes=interval_minutes,
        )


    def _csv_path_failed(
        self,
        symbol: str,
        *,
        interval_minutes: int | None = None,
        bar: int | None = None,
        days_back: int | None = None,
    ) -> Path:
        # build the “good” path exactly as _csv_path would
        orig = self._csv_path(
            symbol,
            interval_minutes=interval_minutes,
            bar=bar,
            days_back=days_back,
        )
        # split into “stem” + “suffix” and append “_failed”
        stem   = orig.stem
        suffix = orig.suffix
        return orig.with_name(f"{stem}_failed{suffix}")


    def _atomic_write(self, df: pd.DataFrame, out_path: Path) -> None:
        atomic_write(df, out_path)
