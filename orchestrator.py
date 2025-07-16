#!/usr/bin/env python3
"""
orchestrator.py — Production‐Grade Pipeline Scheduler (cleaned)
===============================================================

1. Every 1 hour (REFRESH_INTERVAL_HOURS):
   • Ensure all datasets exist (DataManager backfill).
   • Train any coins that aren’t yet live or need refreshed models.
   • Prune model directories so each coin retains ≤ MAX_MODELS_PER_COIN candidates.

2. Every 2 hours (TOURNEY_INTERVAL_HOURS):
   • For coins with ≥ MIN_MODELS, run a tournament to prune to exactly MIN_MODELS (1 champion + challengers).

3. Every 24 hours (PROMOTION_INTERVAL_HOURS):
   • For coins not yet “live” (no live_models/<symbol>_live.json):
       – Backtest the champion for BACKTEST_DAYS.
       – Run size_search → size_configs/<symbol>_best.json.
       – If champion’s backtest return ≥ PROMOTION_THRESHOLD, lock capital via Ledger, write live_models/<symbol>_live.json.

4. Every 30 days (MARKET_TEST_INTERVAL_DAYS):
   • Launch a full sizing‐grid backtest across all coins using pipelines/size_search.py → refresh size_configs/.

A failed coin at any step is recorded in failed_coins.json and logs/status.log.
"""
from __future__ import annotations

import json
import sys
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, List, Optional

import pandas as pd
import yaml
from apscheduler.executors.pool import ThreadPoolExecutor as APS_TPool
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import utils.orch_utils as ou                                                           # :contentReference[oaicite:5]{index=5}
from utils.data_manager import DataManager                                               # :contentReference[oaicite:6]{index=6}
from utils.model_factory import train_and_select, list_models, get_champion               # :contentReference[oaicite:7]{index=7}
from pipelines.tournament import run_tournament                                          # :contentReference[oaicite:8]{index=8}
from pipelines.size_search import search_sizes                                           # :contentReference[oaicite:9]{index=9}
from utils.utils import proxies, stop_flag

# ──────────────────────────────── Constants / Paths ───────────────────────────── #
ROOT         = Path(__file__).resolve().parent
CONFIG_PATH  = ROOT / "config" / "coins.yaml"
LOGS_DIR     = ROOT / "logs"
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
SIZE_DIR     = ROOT / "size_configs"
LIVE_DIR     = ROOT / "live_models"
FAILED_FILE  = ROOT / "failed_coins.json"

# Scheduler intervals
REFRESH_INTERVAL_HOURS       = 1     # backfill + train + prune every 1 hour
TOURNEY_INTERVAL_HOURS       = 2     # tournament prune every 2 hours
PROMOTION_INTERVAL_HOURS     = 24    # promotion pass every 24 hours
MARKET_TEST_INTERVAL_DAYS    = 30    # full sizing grid backtest every 30 days

# Read thresholds from `config/settings.yaml` if present
SETTINGS_PATH = ROOT / "config" / "settings.yaml"
if SETTINGS_PATH.exists():
    SETTINGS = yaml.safe_load(SETTINGS_PATH.read_text()) or {}
else:
    SETTINGS = {}

# Promotion: require ≥ this PnL over BACKTEST_DAYS to go live
PROMOTION_THRESHOLD = float(SETTINGS.get("promotion_threshold", 0.02))    # default 2 % over BACKTEST_DAYS
CAPITAL_PER_MODEL    = float(SETTINGS.get("capital_per_model", 100.0))    # $100 per model allocation

# Pipeline constants
MIN_MODELS           = int(SETTINGS.get("min_models", 4))                     # 1 champ + 3 challengers
BACKTEST_DAYS        = int(SETTINGS.get("backtest_days", 15))                  # 15-day look-back for promotion
MAX_MODELS_PER_COIN  = int(SETTINGS.get("max_models_per_coin", 8))             # prune to top 8 each refresh

# Ensure directories exist
for d in (LOGS_DIR, DATA_DIR, MODELS_DIR, SIZE_DIR, LIVE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────── #
# Initialize logging via orch_utils (JSON or plain)
# ───────────────────────────────────────────────────────────────────────────── #
logger = ou.configure_logging(log_dir=LOGS_DIR)
ou.log.info("Orchestrator starting up…")

# Lower APScheduler’s verbosity so you only see WARN/ERROR from it
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# ───────────────────────────────────────────────────────────────────────────── #
# Thread‐safe proxy updates
# ───────────────────────────────────────────────────────────────────────────── #
_proxies_lock = threading.Lock()
proxies.setdefault("last_action", "idle")
proxies.setdefault("train_jobs", 0)
proxies.setdefault("tourney_runs", 0)
proxies.setdefault("failed_jobs", 0)
proxies.setdefault("live_count", 0)
proxies.setdefault("pending_count", 0)

# Shared state for daily promotion pass
_last_failed: List[str] = []

# ───────────────────────────────────────────────────────────────────────────── #
# Helpers
# ───────────────────────────────────────────────────────────────────────────── #
def _load_coins() -> List[dict]:
    """Load config/coins.yaml → list of coin configs (dicts)."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing coins.yaml at {CONFIG_PATH}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    coins = cfg.get("coins", cfg) if isinstance(cfg, dict) else cfg
    if not isinstance(coins, list):
        raise ValueError("`coins.yaml` must be a list or have top‐level `coins:`")
    return coins

def _record_fail(symbol: str, msg: str) -> None:
    """
    Append an error line to logs/status.log and track the symbol in _last_failed.
    """
    status_log = LOGS_DIR / "status.log"
    timestamp = pd.Timestamp.utcnow().isoformat()
    line = f"{timestamp} | {symbol} | ERROR | {msg}\n"
    with open(status_log, "a", encoding="utf-8") as fh:
        fh.write(line)
    _last_failed.append(symbol)
    with _proxies_lock:
        proxies["failed_jobs"] += 1
        proxies["last_action"] = "error"

def _write_failed_json() -> None:
    """Write the latest _last_failed list to failed_coins.json (overwrites)."""
    if not _last_failed:
        return
    payload = {"failed": sorted(set(_last_failed))}
    with open(FAILED_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    ou.log.warning("Wrote failed_coins.json: %s", payload)

def _safe_run(fn: Callable[[], None]) -> None:
    """
    Simple wrapper to run a job function under try/except and log any exceptions.
    This replaces ou.safe_run(...) to avoid signature mismatches.
    """
    try:
        fn()
    except Exception as exc:
        ou.log.exception("Job %s error: %s", fn.__name__, exc)
        # If you also want to record failures for coins inside each job, that logic already exists 
        # in job_refresh, job_tourney, job_promote, job_market_test via _record_fail().

# ───────────────────────────────────────────────────────────────────────────── #
# Job 1: Refresh Models (backfill datasets, train, prune extra models)
# ───────────────────────────────────────────────────────────────────────────── #
def job_refresh() -> None:
    """
    1. Ensure all datasets exist (DataManager.backfill_if_missing).
    2. For each coin not yet live: train_and_select until up to MAX_MODELS_PER_COIN.
    3. After training, prune older models so that each coin keeps only its best MAX_MODELS_PER_COIN.
    """
    coins = _load_coins()
    # 1. Gather data
    with _proxies_lock:
        proxies["last_action"] = "gather_data"

    dm = DataManager(coins, data_directory=DATA_DIR, verbose=False)
    try:
        dm.ensure_datasets()
    except Exception as exc:
        ou.log.exception("DataManager.ensure_datasets() failed: %s", exc)
        # If data backfill failed entirely, mark all as failed and skip training
        for c in coins:
            _record_fail(c["symbol"], f"Data backfill error: {exc}")
        dm.stop()
        return

    # 2. Train & prune per coin
    for cfg in coins:
        sym = cfg["symbol"]
        live_path = LIVE_DIR / f"{sym}_live.json"
        if live_path.exists():
            # Already live → skip training (we assume champion frozen)
            continue

        try:
            # Load existing model count
            existing = list_models(sym)
            # While #models < MIN_MODELS, train new ones
            while len(existing) < MIN_MODELS:
                with _proxies_lock:
                    proxies["last_action"] = "train"
                    proxies["train_jobs"] += 1
                csv_path = dm._csv_path(sym) if dm._csv_path(sym).exists() else dm._latest_csv(sym)
                if not csv_path or not csv_path.exists():
                    raise RuntimeError(f"No dataset for {sym}")
                train_and_select(cfg, csv_path)
                existing = list_models(sym)

            # Now we have ≥ MIN_MODELS; but we'll allow up to MAX_MODELS_PER_COIN.
            # If > MAX_MODELS_PER_COIN, prune extras via a 7-day tournament
            all_models = list_models(sym)
            if len(all_models) > MAX_MODELS_PER_COIN:
                keep = MAX_MODELS_PER_COIN
                with _proxies_lock:
                    proxies["last_action"] = "prune"
                    proxies["tourney_runs"] += 1
                run_tournament(sym, keep_top=keep)

        except Exception as exc:
            ou.log.exception("[%s] Refresh error: %s", sym, exc)
            _record_fail(sym, f"Refresh error: {exc}")

    dm.stop()

# ───────────────────────────────────────────────────────────────────────────── #
# Job 2: Tournament Prune (every 2 hours)
# ───────────────────────────────────────────────────────────────────────────── #
def job_tourney() -> None:
    """
    For every coin, if it has ≥ MIN_MODELS candidate models, run a 7-day tournament
    to keep exactly MIN_MODELS (1 champ + challengers). Updates survival_stats.
    """
    coins = _load_coins()
    with _proxies_lock:
        proxies["last_action"] = "tourney"
        proxies["tourney_runs"] += 1

    for cfg in coins:
        sym = cfg["symbol"]
        try:
            models_for_sym = list_models(sym)
            if len(models_for_sym) >= MIN_MODELS:
                run_tournament(sym, keep_top=MIN_MODELS)
        except Exception as exc:
            ou.log.exception("[%s] tournament error: %s", sym, exc)
            _record_fail(sym, f"Tournament error: {exc}")

# ───────────────────────────────────────────────────────────────────────────── #
# Job 3: Daily Promotion Pass (every 24 hours)
# ───────────────────────────────────────────────────────────────────────────── #
def job_promote() -> None:
    """
    For each coin not yet live (no live_models/<symbol>_live.json):
      1. Load champion via get_champion.
      2. Backtest last BACKTEST_DAYS candles → subset of dataset CSV.
      3. Run search_sizes → size_info (must include “return”).
      4. If size_info['return'] ≥ PROMOTION_THRESHOLD: lock capital via Ledger, write live_models/<symbol>_live.json.
      5. Otherwise, leave it pending.
    Write any failures to failed_coins.json & status.log.
    """
    coins = _load_coins()
    dm = DataManager(coins, data_directory=DATA_DIR, verbose=False)

    try:
        dm.ensure_datasets()
    except Exception as exc:
        ou.log.exception("DataManager.ensure_datasets() failed in promotion pass: %s", exc)
        # If even the datasets can’t be loaded, skip everything for today
        dm.stop()
        return

    _last_failed.clear()

    for cfg in coins:
        sym = cfg["symbol"]
        live_path = LIVE_DIR / f"{sym}_live.json"
        if live_path.exists():
            # Already live → skip
            continue

        try:
            # 1. Champion
            with _proxies_lock:
                proxies["last_action"] = "load_models"
                proxies["pending_count"] += 1

            champ = get_champion(sym)
            if champ is None:
                raise RuntimeError("No champion found")

            # 2. Backtest
            with _proxies_lock:
                proxies["last_action"] = "backtest"
            csv_path = dm._latest_csv(sym)
            df_all   = pd.read_csv(csv_path)
            cutoff   = pd.Timestamp.utcnow() - pd.Timedelta(days=BACKTEST_DAYS)
            df_recent = df_all[pd.to_datetime(df_all["time"], utc=True) >= cutoff]
            if df_recent.empty:
                raise RuntimeError(f"No data in last {BACKTEST_DAYS} days")

            # 3. Size‐grid search
            with _proxies_lock:
                proxies["last_action"] = "size_search"
            size_info = search_sizes(sym, champ, cfg, df_recent)
            (SIZE_DIR / f"{sym}_best.json").write_text(json.dumps(size_info, indent=2))

            ret15 = size_info.get("return", None)
            if ret15 is None or pd.isna(ret15):
                from utils.utils import quick_score
                ret15 = quick_score(sym, champ) or 0.0

            # 4. Promotion check
            if ret15 >= PROMOTION_THRESHOLD:
                # Lock capital and write live JSON
                with _proxies_lock:
                    proxies["last_action"] = "promote"
                ledger = ou.Ledger()                            # capital management (utils/orch_utils.py) :contentReference[oaicite:10]{index=10}
                locked = ledger.lock(sym, CAPITAL_PER_MODEL)
                if not locked:
                    ou.log.warning("[%s] Not enough capital to lock. Skipping promotion.", sym)
                    continue

                payload = {
                    "champion":    champ.name,
                    "size_info":   size_info,
                    "promoted_at": pd.Timestamp.utcnow().isoformat(),
                }
                live_path.write_text(json.dumps(payload, indent=2))
                with _proxies_lock:
                    proxies["live_count"] = len(list(LIVE_DIR.glob("*_live.json")))

            else:
                ou.log.info("[%s] Champion return %+0.2f < threshold %+0.2f; not promoted.",
                            sym, ret15, PROMOTION_THRESHOLD)

        except Exception as exc:
            ou.log.exception("[%s] Promotion error: %s", sym, exc)
            _record_fail(sym, f"Promotion error: {exc}")

    dm.stop()
    _write_failed_json()

# ───────────────────────────────────────────────────────────────────────────── #
# Job 4: Full Market Test (every 30 days)
# ───────────────────────────────────────────────────────────────────────────── #
def job_market_test() -> None:
    """
    Run a full size‐grid backtest across the entire coin universe to refresh size_configs/<symbol>_best.json.
    Delegates to pipelines/size_search.py’s main() logic if available; otherwise, loops coins individually.
    """
    coins = _load_coins()

    with _proxies_lock:
        proxies["last_action"] = "market_test"

    try:
        # Attempt to call the top‐level entry point if it exists
        import pipelines.size_search as ss_mod
        if hasattr(ss_mod, "main"):
            ss_mod.main()  # e.g. `pipelines/size_search.py --days ...`
            return
    except ImportError:
        pass

    # Fallback: loop coins one by one
    for cfg in coins:
        sym = cfg["symbol"]
        try:
            champ = get_champion(sym)
            if champ is None:
                ou.log.info("[%s] No champion—skipping market_test.", sym)
                continue

            dm = DataManager([cfg], data_directory=DATA_DIR, verbose=False)
            dm.ensure_datasets()
            csv_path = dm._latest_csv(sym)
            df_all   = pd.read_csv(csv_path)
            cutoff   = pd.Timestamp.utcnow() - pd.Timedelta(days=BACKTEST_DAYS)
            df_recent = df_all[pd.to_datetime(df_all["time"], utc=True) >= cutoff]
            if df_recent.empty:
                ou.log.warning("[%s] No recent data—skipping market_test.", sym)
                dm.stop()
                continue

            size_info, full_df = search_sizes(sym, champ, cfg, df_recent)
            (SIZE_DIR / f"{sym}_best.json").write_text(json.dumps(size_info, indent=2))
            dm.stop()

        except Exception as exc:
            ou.log.exception("[%s] market_test error: %s", sym, exc)
            _record_fail(sym, f"market_test error: {exc}")

    _write_failed_json()

# ───────────────────────────────────────────────────────────────────────────── #
# Start the Scheduler
# ───────────────────────────────────────────────────────────────────────────── #
def main() -> None:
    """
    Set up APScheduler with four jobs:
      • job_refresh       → every REFRESH_INTERVAL_HOURS (first run immediately)
      • job_tourney       → every TOURNEY_INTERVAL_HOURS (first run immediately)
      • job_promote       → every PROMOTION_INTERVAL_HOURS (first run immediately)
      • job_market_test   → every MARKET_TEST_INTERVAL_DAYS (first run immediately)

    Press Ctrl‐C to exit cleanly.
    """
    scheduler = BackgroundScheduler(
        executors={"default": APS_TPool(max_workers=8)},
        job_defaults={"coalesce": False, "max_instances": 1},
        timezone="UTC",
    )

    # 1. Refresh Models: every hour, but run once immediately at startup
    scheduler.add_job(
        lambda: _safe_run(job_refresh),
        trigger=IntervalTrigger(hours=REFRESH_INTERVAL_HOURS),
        next_run_time=datetime.utcnow(),
        id="refresh_models",
    )

    # 2. Tournament Prune: every 2 hours, but run once immediately
    scheduler.add_job(
        lambda: _safe_run(job_tourney),
        trigger=IntervalTrigger(hours=TOURNEY_INTERVAL_HOURS),
        next_run_time=datetime.utcnow(),
        id="tourney_prune",
    )

    # 3. Daily Promotion Pass: every 24 hours, but run once immediately
    scheduler.add_job(
        lambda: _safe_run(job_promote),
        trigger=IntervalTrigger(hours=PROMOTION_INTERVAL_HOURS),
        next_run_time=datetime.utcnow(),
        id="daily_promote",
    )

    # 4. Full Market Test: every 30 days, but run once immediately
    scheduler.add_job(
        lambda: _safe_run(job_market_test),
        trigger=IntervalTrigger(days=MARKET_TEST_INTERVAL_DAYS),
        next_run_time=datetime.utcnow(),
        id="market_test",
    )

    scheduler.start()
    ou.log.info("Scheduler started. Jobs: %s", scheduler.get_jobs())

    try:
        # Keep alive until Ctrl‐C
        while not stop_flag.is_set():
            scheduler._event.wait(1)
    except (KeyboardInterrupt, SystemExit):
        ou.log.info("Shutdown signal received. Shutting down scheduler…")
        scheduler.shutdown(wait=False)

    ou.log.info("Orchestrator exiting. Goodbye.")


if __name__ == "__main__":
    main()
