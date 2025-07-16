# """
# main.py – lightweight orchestrator for continuous model evolution & market‑testing
# ===============================================================================

# This script glues together the existing *build & train*, *tournament pruning* and
# *grid‑search market‑test* pipelines into one long‑running process.  Everything is
# scheduled with ***APScheduler*** so you may run the file once and simply keep it
# alive – Cron‑style scheduling is handled internally.

# Key responsibilities
# --------------------
# 1. **Model training** – refresh / (re)train the population for every coin.
# 2. **Leaderboard refresh** – hourly tournament keeping only the best **four**
#    models (1 champion + 3 challengers).
# 3. **Market‑test grid search** – every two months run the sizing‑grid back‑test
#    to discover the best position‑sizing profile per coin.  The resulting JSON
#    files are automatically picked up by *pipelines/trading.py* at runtime.
# 4. **Logging / safety** – rich log output, graceful shutdown on SIGINT/SIGTERM
#    and extensive exception isolation so a failure in one job never stops the
#    whole orchestrator.

# The implementation is intentionally *thin*: all heavy lifting is delegated to
# modules that already exist in the code‑base (`utils.model_factory`,
# `pipelines.tournament`, `pipelines.market_test`, …).  The new helper utilities
# live in *utils/orch_utils.py* – see that file for small but reusable helpers.
# """
# from __future__ import annotations

# import logging
# import signal
# import sys
# import threading
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import List

# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.triggers.interval import IntervalTrigger

# # ───────────────────────── internal imports ──────────────────────────── #
# # NOTE: importing inside functions where the modules are heavy or carry
# #       non‑trivial side‑effects (e.g. torch) to keep startup snappy.
# from utils.orch_utils import configure_logging, load_coins_cfg, safe_call

# # absolute project paths -------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent
# LOGS_DIR = BASE_DIR / "logs"
# DATA_DIR = BASE_DIR / "data"

# # schedule cadence -------------------------------------------------------
# TRAIN_INTERVAL_H = 12        # re‑train twice per day
# TOURNEY_INTERVAL_H = 1       # prune leaderboard hourly
# MKTTEST_INTERVAL_D = 60      # grid‑search every ~2 months

# # thread‑pool sizing -----------------------------------------------------
# MAX_THREADS = 32             # hard upper‑limit – adjust to host

# # global stop‑flag – shared by signal handler & scheduler ---------------
# _stop_event = threading.Event()


# # ══════════════════════════ scheduled jobs ══════════════════════════════ #

# def job_train_models(coins: List[dict]):
#     """(Re)train and auto‑select best model(s) for every coin."""
#     from utils.model_factory import train_and_select  # heavy import
#     from utils.data_manager import DataManager

#     logger = logging.getLogger("job.train")
#     dm = DataManager(coins, data_directory=DATA_DIR, verbose=False)
#     # ensure the datasets are fresh before training
#     dm.backfill_if_missing()

#     with ThreadPoolExecutor(max_workers=min(len(coins), MAX_THREADS)) as pool:
#         futs = {
#             pool.submit(train_and_select, cfg, dm._csv_path(cfg["symbol"])): cfg
#             for cfg in coins
#         }
#         for fut in as_completed(futs):
#             cfg = futs[fut]
#             sym = cfg["symbol"]
#             try:
#                 fut.result()
#                 logger.info("[%s] training finished", sym)
#             except Exception as exc:  # noqa: BLE001
#                 logger.exception("[%s] training failed: %s", sym, exc)
#     dm.stop()


# def job_tournament(coins: List[dict]):
#     """Run hourly tournament pruning to maintain champion + 3 challengers."""
#     from pipelines.tournament import run_tournament  # light import

#     logger = logging.getLogger("job.tourney")
#     for cfg in coins:
#         sym = cfg["symbol"]
#         try:
#             run_tournament(sym, keep_top=4)
#             logger.info("[%s] tournament complete", sym)
#         except Exception as exc:  # noqa: BLE001
#             logger.exception("[%s] tournament error: %s", sym, exc)


# def job_market_test():
#     """Launch the grid‑search market‑test (runs ~minutes‑hours)."""
#     import importlib

#     logger = logging.getLogger("job.market_test")
#     try:
#         # *market_test.main()* consumes argparse, so we create a fresh
#         # module instance to avoid clobbering global sys.argv state across
#         # scheduler runs.
#         mt = importlib.import_module("market_test")
#         mt.main()  # uses defaults: 30‑day window / 1,000 combos
#         logger.info("market‑test grid search finished")
#     except SystemExit as exc:  # suppress argparse sys.exit()
#         # argparse in *market_test* calls sys.exit(0) on success
#         if exc.code not in (0, None):
#             logger.error("market‑test exited with code %s", exc.code)
#     except Exception as exc:  # noqa: BLE001
#         logger.exception("market‑test failed: %s", exc)


# # ═══════════════════════ graceful shutdown ═════════════════════════════ #

# def _sig_handler(signum, _):
#     logging.getLogger("main").warning("received %s – initiating shutdown", signal.Signals(signum).name)
#     _stop_event.set()


# # ═════════════════════════════ main() ══════════════════════════════════ #

# def main() -> None:  # noqa: C901 – a bit long but self‑contained
#     configure_logging(LOGS_DIR)
#     logger = logging.getLogger("main")

#     # wire Ctrl‑C & SIGTERM so docker / systemd can stop us cleanly
#     signal.signal(signal.SIGINT, _sig_handler)
#     if hasattr(signal, "SIGTERM"):
#         signal.signal(signal.SIGTERM, _sig_handler)

#     coins = load_coins_cfg(Path("config/coins.yaml"))
#     if not coins:
#         logger.error("coins.yaml produced zero coins – nothing to do")
#         sys.exit(1)
#     logger.info("loaded %d coins: %s", len(coins), ", ".join(c["symbol"] for c in coins))

#     # scheduler + thread‑pool context
#     with BackgroundScheduler(daemon=True) as sched:
#         # spread the heavier tasks so they don’t all fire at once after restart
#         sched.add_job(safe_call(job_train_models),    IntervalTrigger(hours=TRAIN_INTERVAL_H), args=[coins], id="train")
#         sched.add_job(safe_call(job_tournament),      IntervalTrigger(hours=TOURNEY_INTERVAL_H), args=[coins], id="tourney")
#         sched.add_job(safe_call(job_market_test),     IntervalTrigger(days=MKTTEST_INTERVAL_D), id="mkt_test")

#         sched.start()
#         logger.info("scheduler started – press Ctrl‑C to stop")

#         # keep‑alive loop -------------------------------------------------
#         try:
#             while not _stop_event.is_set():
#                 time.sleep(1)
#         finally:
#             logger.info("shutting down scheduler…")
#             sched.shutdown(wait=False)
#     logger.info("main.py exit – goodbye")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""main.py – Continuous model‑evolution orchestrator
====================================================

Runs three independent cron jobs using APScheduler:

1. **refresh_models_job** – builds new models from the latest data and prunes to
   the best 25 via backtests.  Runs every *REFRESH_INTERVAL_HOURS* (default 1 h).
2. **promote_job** – if ≥4 contender models exist, launches a tournament and,
   provided the champion beats a configurable performance threshold, promotes it
   to live trading with a $25 (or any available balance) stake for two weeks.
   Runs every 30 minutes.
3. **market_test_job** – every 60 days runs an expensive grid‑sizing market test
   across the full coin universe to fine‑tune model sizing.

All jobs are wrapped in **orch_utils.safe_run** so an exception in one task will
never halt the scheduler.  Execution progress is logged to *stdout* and a
rotating file under **./logs**.
"""
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

import utils.orch_utils as ou

# ---------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------
CFG_PATH = Path("config") / "settings.yaml"
COINS_PATH = Path("config") / "coins.yaml"
CAPITAL_PATH = Path(".orchestrator") / "capital.yaml"
STATE_PATH = Path(".orchestrator") / "state.yaml"

logger = ou.configure_logging()
cfg = ou.load_yaml(CFG_PATH)
coins_cfg = ou.load_yaml(COINS_PATH).get("coins", [])

MODELS_DIR = Path(cfg.get("models_dir", "models"))
PROMOTION_THRESHOLD = float(cfg.get("promotion_threshold", 0.01))  # 1 % per backtest window
CAPITAL_PER_MODEL = float(cfg.get("capital_per_model", 25))
REFRESH_INTERVAL_HOURS = int(cfg.get("refresh_interval_hours", 1))
MARKET_TEST_INTERVAL_DAYS = int(cfg.get("market_test_interval_days", 60))

# ---------------------------------------------------------------------------
# Capital helpers -------------------------------------------------------------

def read_capital_pool() -> float:
    if not CAPITAL_PATH.exists():
        return 0.0
    return yaml.safe_load(CAPITAL_PATH.read_text()).get("available", 0.0)


def write_capital_pool(available: float):
    CAPITAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAPITAL_PATH.write_text(yaml.safe_dump({"available": available}))


# ---------------------------------------------------------------------------
# Scheduler jobs --------------------------------------------------------------

def refresh_models_job():
    """Build new candidate models and prune to top 25."""
    import trading_test, modelsaving_test  # user‑supplied helpers

    logger.info("[REFRESH] Building new candidate models…")
    trading_test.build_models()

    logger.info("[REFRESH] Pruning candidate pool to 25 best backtesters…")
    modelsaving_test.prune_models(max_keep=25, models_dir=MODELS_DIR)

    logger.info("[REFRESH] Done.")


def promote_job():
    """Run a four‑way tournament and, if champion clears threshold, go live."""
    logger.info("[PROMOTE] Checking promotion eligibility…")

    models_meta = ou.discover_models(MODELS_DIR)
    if len(models_meta) < 4:
        logger.info("[PROMOTE] Need ≥4 contenders – currently %d. Skipping.", len(models_meta))
        return

    competitors = ou.choose_competitors(models_meta, top_n=4)
    winner_path, perf = ou.run_tournament(competitors, logger)

    if not ou.meets_threshold(perf, PROMOTION_THRESHOLD):
        logger.info("[PROMOTE] Champion %.4f < threshold %.4f. Skipping promotion.", perf, PROMOTION_THRESHOLD)
        return

    capital_pool = read_capital_pool()
    allocation = ou.allocate_capital(capital_pool, CAPITAL_PER_MODEL)
    if allocation <= 0:
        logger.warning("[PROMOTE] No capital available (%.2f). Skipping promotion.", capital_pool)
        return

    if ou.promote_to_live(winner_path, allocation, logger):
        write_capital_pool(capital_pool - allocation)
        _record_live_model(winner_path, allocation)


def _record_live_model(path: Path, alloc: float):
    state = ou.load_yaml(STATE_PATH)
    live = state.get("live_models", [])
    live.append({"model": str(path), "capital": alloc, "started": datetime.utcnow().isoformat()})
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(yaml.safe_dump(state | {"live_models": live}))


def market_test_job():
    """Heavy grid‑search market test every **MARKET_TEST_INTERVAL_DAYS**."""
    state = ou.load_yaml(STATE_PATH)
    last = datetime.fromisoformat(state.get("last_market_test", "1970-01-01T00:00:00"))
    if datetime.utcnow() - last < timedelta(days=MARKET_TEST_INTERVAL_DAYS):
        logger.debug("[MTEST] Not due yet. Next in %s.", timedelta(days=MARKET_TEST_INTERVAL_DAYS) - (datetime.utcnow() - last))
        return

    logger.info("[MTEST] Launching market‑test …")
    models_meta = ou.discover_models(MODELS_DIR)
    ou.run_market_test(models_meta, coins_cfg, logger)

    STATE_PATH.write_text(yaml.safe_dump(state | {"last_market_test": datetime.utcnow().isoformat()}))
    logger.info("[MTEST] Done.")


# ---------------------------------------------------------------------------
# Entry‑point -----------------------------------------------------------------

def main():
    scheduler = BlockingScheduler(
        executors={"default": ThreadPoolExecutor(max_workers=3)},
        job_defaults={"misfire_grace_time": 300},
    )

    scheduler.add_job(lambda: ou.safe_run(logger, refresh_models_job), "interval", hours=REFRESH_INTERVAL_HOURS, next_run_time=datetime.utcnow())
    scheduler.add_job(lambda: ou.safe_run(logger, promote_job), "interval", minutes=30, next_run_time=datetime.utcnow() + timedelta(minutes=5))
    scheduler.add_job(lambda: ou.safe_run(logger, market_test_job), "cron", hour=2, minute=0)

    logger.info("Scheduler initialised – Ctrl‑C to quit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Graceful shutdown…")


if __name__ == "__main__":
    main()
