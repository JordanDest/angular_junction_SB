"""
tournament.py ­– model promotion / pruning logic
────────────────────────────────────────────────

Maintains one *champion* and three *challengers* per symbol.

• Evaluates every model over the last 7 days of candles.
• Keeps the `keep_top` best; prunes all others.
• Appends every evaluation round to *tournament_leaderboard.csv*.

This module is used two ways:

  ① `main.py` calls :pyfunc:`run_tournament` ad‑hoc from APScheduler
  ② You can run the file directly for a manual or continuous tournament:

       python -m pipelines.tournament once     # single pass
       python -m pipelines.tournament          # hourly forever
"""

from __future__ import annotations

import csv, time, sys, yaml
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils.data_manager import DataManager
from utils.model_factory import list_models, get_next_candidate
from core.ktrader import KTrader
from utils.utils import configure_logging, get_logger, stop_flag, proxies, survival_stats

# ───────────────────────── configuration ────────────────────────────── #
MAX_WORKERS            = 25
ROLLING_WINDOW_MINUTES = 120            # for utils.quick_score (not used here)
EVAL_WINDOW_DAYS       = 7
CHALLENGERS_PER_SYMBOL = 3              # kept for future “ladder” mode
REPLACEMENT_THRESHOLD  = Decimal("0.10")   # unused in one‑shot mode
LEADERBOARD            = Path("tournament_leaderboard.csv")

METRIC_FIELDS = [
    "ts", "symbol", "model_id", "role",
    "trades", "pnl", "avg_gain", "win_rate",
]

# ───────────────────────── logging setup ─────────────────────────────── #
configure_logging()
log = get_logger("Tournament")

# ───────────────────────── metric helper ─────────────────────────────── #
def _evaluate(symbol: str, model_path: Path,
              candles: pd.DataFrame, cfg: dict) -> Dict[str, Any]:
    """Full back‑test of *model_path* over *candles*; returns PnL metrics."""
    if stop_flag.is_set():              # honour Ctrl‑C / SIGTERM fast
        raise RuntimeError("Stop‑flag set – abort evaluation")

    trader = KTrader(cfg, model_path, global_equity=[100.0])
    for row in candles.itertuples(index=False):
        trader.step(row._asdict())
        if stop_flag.is_set():
            raise RuntimeError("Stop‑flag set – abort evaluation")

    trades_df = pd.DataFrame(trader.trade_log)
    pnl       = trader.capital - 100.0
    trades    = len(trades_df)
    win_rate  = float((trades_df["pnl"] > 0).mean()) if trades else 0.0
    avg_gain  = float(trades_df["pnl"].mean())       if trades else 0.0

    return {"pnl": pnl, "trades": trades,
            "win_rate": win_rate, "avg_gain": avg_gain}

# ───────────────────────── one‑shot API (called by main.py) ──────────── #
def run_tournament(symbol: str, keep_top: int = 4) -> None:
    """Rank *all* models for *symbol*, keep `keep_top`, prune the rest."""
    if stop_flag.is_set():
        return                                           # graceful skip

    models = list_models(symbol)
    if len(models) <= keep_top:
        return                                           # nothing to prune

    # coin config is stored in utils.utils.COINS (injected by main.py)
    try:
        from utils.utils import COINS                    # injected list
        coin_cfg = next(c for c in COINS if c["symbol"] == symbol)
    except (ImportError, StopIteration):
        log.warning("%s: coin config missing – skipping", symbol)
        return

    dm        = DataManager([coin_cfg])
    csv_path  = dm._csv_path(symbol)

    # fallback to latest available CSV if today’s file not present
    if not csv_path.exists():
        csv_path = dm._latest_csv(symbol)
    if not csv_path or not csv_path.exists():
        log.warning("%s: no CSV dataset yet", symbol)
        return

    df_full = pd.read_csv(csv_path)
    cutoff  = datetime.now(timezone.utc) - timedelta(days=EVAL_WINDOW_DAYS)
    df_eval = df_full[pd.to_datetime(df_full["time"]) >= cutoff]
    if df_eval.empty:
        log.warning("%s: not enough recent data", symbol)
        return

    results: List[Tuple[float, Path, Dict[str, Any]]] = []

    max_workers = min(MAX_WORKERS, len(models))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eval") as tp:
        fut = {tp.submit(_evaluate, symbol, mp, df_eval, coin_cfg): mp for mp in models}
        for f in as_completed(fut):
            mp = fut[f]
            if stop_flag.is_set():
                tp.shutdown(wait=False, cancel_futures=True)
                return
            try:
                m = f.result()
                results.append((m["pnl"], mp, m))
                log.info("%s | %s → pnl %.2f trades %d",
                         symbol, mp.stem, m["pnl"], m["trades"])
            except Exception as exc:                      # noqa: BLE001
                log.warning("%s: eval failed for %s (%s)", symbol, mp.name, exc)

    if not results:
        return

    # ── keep the best N ──────────────────────────────────────────────── #
    results.sort(key=lambda t: t[0], reverse=True)
    top   = results[:keep_top]
    prune = results[keep_top:]

    # ── write leaderboard snapshot ───────────────────────────────────── #
    header_needed = not LEADERBOARD.exists()
    with LEADERBOARD.open("a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(METRIC_FIELDS)
        ts = datetime.now(timezone.utc).isoformat()
        for pnl, mp, m in results:
            role = "kept" if (pnl, mp, m) in top else "pruned"
            w.writerow([ts, symbol, mp.stem, role,
                        m["trades"], pnl, m["avg_gain"], m["win_rate"]])

    # ── prune losers from disk ───────────────────────────────────────── #
    for _, path, _ in prune:
        path.unlink(missing_ok=True)
        path.with_name(path.stem + "_sc.pkl").unlink(missing_ok=True)
        log.info("%s: pruned %s", symbol, path.name)


    # ── dashboard bookkeeping ────────────────────────────────────────── #
    proxies["tourney_runs"] += 1
    proxies["last_action"]   = "tourney"

    # model‑variant survival counts (based on the *kept* models)
    variant_counts: Dict[str, int] = {}
    for _, mp, _ in top:
        # file name pattern assumed: <symbol>_<variant>.pkl
        variant = "_".join(mp.stem.split("_")[1:]) or mp.stem
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
    survival_stats[symbol] = variant_counts

    
# ───────────────────────── optional forever‑runner (CLI) ─────────────── #
class TournamentManager:
    """Hourly tournament loop (CLI / stand‑alone use, not called by main.py)."""
    def __init__(self, coins_cfg: List[dict]):
        self.coins_cfg = coins_cfg
        self.dm = DataManager(coins_cfg)
        self.dm.start(refresh_days=30)

    def run_forever(self):
        try:
            while not stop_flag.is_set():
                start = time.perf_counter()
                for c in self.coins_cfg:
                    run_tournament(c["symbol"], keep_top=4)
                    if stop_flag.is_set():
                        break
                elapsed = time.perf_counter() - start
                sleep_for = max(0, 3_600 - elapsed)
                time.sleep(sleep_for)
        finally:
            self.dm.stop()
            log.info("TournamentManager stopped")

# ───────────────────────────── CLI entrypoint ────────────────────────── #
if __name__ == "__main__":
    coins_cfg = yaml.safe_load(Path("config/coins.yaml").read_text())["coins"]

    if len(sys.argv) > 1 and sys.argv[1] == "once":
        for c in coins_cfg:
            run_tournament(c["symbol"], keep_top=4)
    else:
        TournamentManager(coins_cfg).run_forever()
