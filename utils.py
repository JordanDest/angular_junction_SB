# utils.py
"""Common helpers used across *StockFactory*.

This module deliberately contains **zero third‑party imports** so it can be
imported early (even before the virtual‑env is active).

Functions
~~~~~~~~~
* ``safe_convert`` – tolerant casting with fallback.
* ``round_down`` / ``round_up`` – fixed‑decimal rounding for exchange tick‑sizes.
* ``configure_logging`` – JSON‑lines logger ready for fast tail‑ing.
* ``get_logger`` – child loggers with consistent JSON structure.
* ``log_trade`` – *thread‑safe* CSV append & dict echo for immediate use.
* ``heartbeat`` – one‑liner status ping (logs + optional notifier).
* ``quick_score`` – lightweight forward‑walk PnL scorer (used by tournament).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
import pandas as pd
import os

# ------------------------------------------------------------------ #
# module‑wide constants & logger
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parent.parent      # project root
_LOG      = logging.getLogger("Utils")
###############################################################################
# tolerant casting
###############################################################################


def safe_convert(value: Any, target_type: type, default: Any | None = None) -> Any:
    """Safely convert *value* to *target_type*.

    *None*, *NaN*, or conversion errors return *default* instead of raising.
    """
    if value is None:
        return default
    try:
        # Avoid "ValueError: cannot convert float NaN to integer" for ints.
        if isinstance(value, float) and math.isnan(value):
            return default
        return target_type(value)
    except (ValueError, TypeError):
        return default


###############################################################################
# math helpers
###############################################################################


def round_down(x: float, decimals: int) -> float:
    factor: float = 10 ** decimals
    return math.floor(x * factor) / factor


def round_up(x: float, decimals: int) -> float:
    factor: float = 10 ** decimals
    return math.ceil(x * factor) / factor

# ------------------------------------------------------------------ #
# internal helpers (kept private – not exported)
# ------------------------------------------------------------------ #

def _locate_latest_csv(symbol: str, data_dir: Path) -> Path | None:
    """
    Return the freshest OHLC CSV file produced by :pyclass:`DataManager`
    for *symbol*.  Pattern: ``{symbol}_<days>d_<intv>m_<date>.csv``.
    """
    files = sorted(
        data_dir.glob(f"{symbol}_*d_*m_*.csv"),
        key=os.path.getmtime,
        reverse=True,    )
    return files[0] if files else None
###############################################################################
# logging helpers – JSON‑lines for easy ingestion
###############################################################################

_JSON_LOCK = threading.Lock()


class _JsonFormatter(logging.Formatter):
    """Very lightweight JSON formatter suitable for millions of lines/day."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "name": record.name,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        with _JSON_LOCK:
            return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logger exactly **once** (subsequent calls are no‑ops)."""
    root = logging.getLogger()
    if getattr(root, "_configured", False):  # type: ignore[attr-defined]
        return  # already done

    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Short‑hand for *consistent* child loggers."""
    return logging.getLogger(name)


###############################################################################
# trade‑logging helpers
###############################################################################

_TRADE_LOCK = threading.Lock()
_TRADE_FN = Path("trade_log.csv")
_TRADE_HEADER = [
    "ts",
    "symbol",
    "side",
    "qty",
    "price",
    "mode",
    "cap_before",
    "cap_after",
    "pnl",
]


def log_trade(
    ts: str | datetime,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    mode: str,
    cap_before: float,
    cap_after: float,
    pnl: float,
) -> Dict[str, Any]:
    """Append one trade to *trade_log.csv* in a thread‑safe way.

    Returns the trade as a plain ``dict`` (handy for in‑memory use & tests).
    """
    ts_str = ts.isoformat() if isinstance(ts, datetime) else ts

    row = {
        "ts": ts_str,
        "symbol": symbol,
        "side": side,
        "qty": round(qty, 6),
        "price": round(price, 6),
        "mode": mode,
        "cap_before": round(cap_before, 2),
        "cap_after": round(cap_after, 2),
        "pnl": round(pnl, 2),
    }

    with _TRADE_LOCK:
        need_header = not _TRADE_FN.exists()
        with _TRADE_FN.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADE_HEADER)
            if need_header:
                writer.writeheader()
            writer.writerow(row)
    return row


###############################################################################
# heartbeat helper (re‑usable by trading & live_trading)
###############################################################################


def heartbeat(mode: str, pair: str, equity: float, pos_units: float) -> None:
    """Log/notify a short *still‑alive* status line."""
    logger = get_logger("Heartbeat")
    logger.info(
        "HB %s %s – equity %.2f pos %.6f", mode.upper(), pair, equity, pos_units
    )
    # Optional notifier integration (safe import).
    try:
        from feedback.texting import notify  # type: ignore

        notify(
            f"[{mode}] HB {pair} | Equity {equity:.2f} | Pos {pos_units:.6f} @ "
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}"
        )
    except ModuleNotFoundError:
        pass  # texting not configured – no‑op


###############################################################################
# lightweight forward tester (used by tournament.py)
###############################################################################

ROLLING: Dict[str, Deque[Any]] = {}
CAPITAL: float = 100.0
COINS: List[Dict[str, Any]] = []


def quick_score(
    symbol: str,
    model_path: Path,
    *,
    csv_tail: int = 1_500,
    starting_capital: float = 100.0,
    data_directory: Path | str = BASE_DIR / "data",
) -> float | None:
    """
    **Very** fast proxy‑score used inside the grid‑search:

    1.  grabs the newest OHLC CSV on disk,
    2.  walks only the last *csv_tail* rows through *KTrader*,
    3.  returns *Δ capital* (positive = profitable).
    """
    from core.ktrader import KTrader            # lazy import – avoids cycles

    csv_path = _locate_latest_csv(symbol, Path(data_directory))
    if csv_path is None:
        _LOG.warning("quick_score: dataset for %s not found in %s", symbol, data_directory)
        return None

    df = pd.read_csv(csv_path).sort_values("time").tail(csv_tail)
    if len(df) < 30:        # LSTM needs a warm‑up window
        return None

    trader = KTrader({"symbol": symbol}, model_path, [starting_capital])
    for bar in df.to_dict("records"):
        trader.step(bar)

    if getattr(trader, "pos_units", 0) > 0:
        last_mid = safe_convert(df.iloc[-1].get("mid"), float, None)
        if last_mid is None:             # should never happen, but be safe
           last_mid = (
                safe_convert(df.iloc[-1].get("high"), float, 0.0)
                + safe_convert(df.iloc[-1].get("low"),  float, 0.0)
            ) / 2
        trader.liquidate(last_mid)


    return trader.capital - starting_capital


# ------------------------------------------------------------------ #
# Extended metric – optional but requested: *thorough_score*
# ------------------------------------------------------------------ #

def thorough_score(
    symbol: str,
    model_path: Path,
    *,
    starting_capital: float = 100.0,
    days_back: int = 5,
    data_directory: Path | str = BASE_DIR / "data",
) -> float | None:
    """
    “Paper walk” all the way **from the last candle in the stored CSV to *now*,
    filling the gap with live trades** fetched via the Kraken ``/Trades`` API.

    The live portion is held only in‑memory – the on‑disk dataset is *not*
    mutated.  Useful for a more realistic sanity‑check before promoting a
    model to live trading.
    """
    try:
        from utils.data_manager import (               # local import – OK
            _aggregate_trades,
            _KrakenAccessor,
            OHLC_COLUMNS,
        )
        from kraken_api import KrakenREST             # project local
    except Exception as exc:
        _LOG.warning("thorough_score: import failure (%s)", exc)
        return None

    csv_path = _locate_latest_csv(symbol, Path(data_directory))
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    # ------------------------------------------------------------------ #
    # pull fresh trades to cover gap → aggregate → append
    # ------------------------------------------------------------------ #
    last_ts = int(pd.to_datetime(df["time"].iloc[-1], utc=True).timestamp())
    accessor = _KrakenAccessor(KrakenREST())
    try:
        trades, _ = accessor.trades(pair=symbol, since=last_ts * 1_000_000_000)
    except Exception as exc:
        _LOG.warning("thorough_score: Kraken fetch failed (%s)", exc)
        trades = []

    if trades:
        new_bars = _aggregate_trades(trades, 60)   # DataManager default = 60 s
        if new_bars:
            df_extra = pd.DataFrame(new_bars, columns=OHLC_COLUMNS)
            df = (
                pd.concat([df, df_extra], ignore_index=True)
                  .drop_duplicates(subset="time", keep="last")
                  .sort_values("time")
            )

    # ------------------------------------------------------------------ #
    # run the merged frame through KTrader
    # ------------------------------------------------------------------ #
    from core.ktrader import KTrader

    trader = KTrader({"symbol": symbol}, model_path, [starting_capital])
    for bar in df.to_dict("records"):
        trader.step(bar)
    if getattr(trader, "pos_units", 0) > 0:
        last_mid = safe_convert(df.iloc[-1].get("mid"), float, None)
        if last_mid is None:             # should never happen, but be safe
           last_mid = (
                safe_convert(df.iloc[-1].get("high"), float, 0.0)
                + safe_convert(df.iloc[-1].get("low"),  float, 0.0)
            ) / 2
        trader.liquidate(last_mid)

    return trader.capital - starting_capital


stop_flag = threading.Event()

# Global counters / gauges that every module imports and mutates.
proxies: Dict[str, Any] = {
    "train_jobs":       0,
    "trading_threads":  0,
    "tourney_runs":     0,
    "last_action":     "idle",
    "cpu_percent":      0.0,
    "ram_percent":      0.0,
}

# Tournament‑survival statistics by symbol
survival_stats: Dict[str, Any] = {}

# `main.py` will inject the loaded coins list here so that
# `pipelines.tournament` can grab it without a direct dependency.
COINS: list = []