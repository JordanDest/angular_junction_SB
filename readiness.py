# #!/usr/bin/env python3
# """readiness_check.py – end‑to‑end readiness loop
# =================================================

# Runs the **full pipeline per coin** until *all* coins are processed, then
# stages a cross‑coin tournament to pick the *best* contenders for live
# tracking.

# * Steps per coin
#   1. Make sure the minute‑bar CSV is present (DataManager back‑fill).
#   2. Train until **≥4 models** exist (one champion + three challengers).
#   3. Run the 7‑day *tournament* to prune and freeze the top 4.
#   4. 15‑day back‑test to obtain headline metrics.
#   5. Search the position‑sizing grid (``size_search.search_sizes``) and save
#      the best overrides to `size_configs/<symbol>_best.json`.

# * After every coin has a summary row the script:
#   * Sorts by back‑test return and prints the league table.
#   * Picks the **top N (default 5)** coins, re‑runs a *one‑off* tournament over
#     their champions to sanity‑check consistency and prints the podium.
#   * (Optional – see TODO) Locks real capital in the `orch_utils.Ledger` and
#     calls a future ``deploy_live()`` to spawn the live `KTrader` process.

# The loop is *idempotent*: re‑running it will touch only the stale or missing
# parts and skip fresh artefacts.  Typical runtime for ~8 coins on a modern
# laptop is <5 minutes with Torch on CPU.
# """
# from __future__ import annotations

# import json
# import logging
# import os
# import sys
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Dict, List

# import pandas as pd
# import yaml

# from utils.data_manager import DataManager
# from utils.model_factory import get_champion, list_models, train_and_select
# from pipelines.tournament import run_tournament
# from pipelines.size_search import search_sizes  # local grid‑search helper
# from utils.utils import configure_logging, get_logger, stop_flag

# # ───────────────────────── static paths ──────────────────────────────── #
# ROOT = Path(__file__).resolve().parents[1]  # StockFactory2/
# CONFIG_YAML = ROOT / "config" / "coins.yaml"
# DATA_DIR = ROOT / "data"
# MODELS_DIR = ROOT / "models"
# SIZE_DIR = ROOT / "size_configs"
# LOGS_DIR = ROOT / "logs"

# # ───────────────────────── parameters ────────────────────────────────── #
# MIN_MODELS = 4            # champion + 3 challengers
# BACKTEST_DAYS = 15        # look‑back window for quick readiness score
# TOP_N_LIVE = 5            # contenders forwarded to live promotion step
# THREADS = min(os.cpu_count() or 8, 16)

# SIZE_DIR.mkdir(exist_ok=True)
# LOGS_DIR.mkdir(exist_ok=True)

# configure_logging(LOGS_DIR)
# log = get_logger("readiness")

# # ----------------------------------------------------------------------
# # YAML loader helpers
# # ----------------------------------------------------------------------

# def _load_coins(path: Path = CONFIG_YAML) -> Dict[str, dict]:
#     if not path.exists():
#         raise FileNotFoundError(f"coins.yaml missing at {path}")
#     data = yaml.safe_load(path.read_text())
#     coins = data.get("coins", data) if isinstance(data, dict) else data
#     if not isinstance(coins, list):
#         raise ValueError("`coins.yaml` expects a list or a top‑level `coins:` key")
#     return {c["symbol"]: c for c in coins}

# # ----------------------------------------------------------------------
# # Model helpers
# # ----------------------------------------------------------------------

# def _ensure_models(cfg: dict, dm: DataManager) -> None:
#     """Block until *cfg["symbol"]* has at least ``MIN_MODELS`` models on disk."""
#     symbol = cfg["symbol"]
#     csv_path = dm._csv_path(symbol)
#     if not csv_path.exists():
#         csv_path = dm._latest_csv(symbol)
#     if not csv_path or not csv_path.exists():
#         raise RuntimeError(f"Dataset for {symbol} missing – call DataManager.ensure_datasets() first")

#     # Re‑train until the model count reaches MIN_MODELS
#     while not stop_flag.is_set() and len(list_models(symbol)) < MIN_MODELS:
#         log.info("[%s] training (need %d models)…", symbol, MIN_MODELS)
#         train_and_select(cfg, csv_path)

#     # Final prune to exactly MIN_MODELS (1 champ + challengers)
#     run_tournament(symbol, keep_top=MIN_MODELS)

# # ----------------------------------------------------------------------
# # Per‑coin pipeline
# # ----------------------------------------------------------------------

# def _process_coin(cfg: dict, dm: DataManager) -> dict:
#     symbol = cfg["symbol"]
#     _ensure_models(cfg, dm)

#     champ = get_champion(symbol)
#     if champ is None:
#         raise RuntimeError(f"Champion discovery failed for {symbol}")

#     # 15‑day back‑test for readiness score
#     csv_path = dm._latest_csv(symbol)
#     df = pd.read_csv(csv_path)
#     cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=BACKTEST_DAYS)
#     df = df[pd.to_datetime(df["time"], utc=True) >= cutoff]
#     if df.empty:
#         raise RuntimeError(f"No recent data for {symbol}")

#     best_cfg = search_sizes(symbol, champ, cfg, df)
#     (SIZE_DIR / f"{symbol}_best.json").write_text(json.dumps(best_cfg, indent=2))

#     # Quick metrics
#     pnl = best_cfg.get("return", float('nan'))  # search_sizes returns dict w/o perf; fallback below
#     if pd.isna(pnl):
#         from utils.utils import quick_score
#         pnl = quick_score(symbol, champ) or 0.0
#     return {
#         "coin": symbol,
#         "models": len(list_models(symbol)),
#         "champion": champ.name,
#         "return": pnl,
#     }

# # ----------------------------------------------------------------------
# # Cross‑coin league table & live promotion stub
# # ----------------------------------------------------------------------

# def _global_playoffs(summary: pd.DataFrame) -> None:
#     table = summary.sort_values("return", ascending=False).reset_index(drop=True)
#     print("\n🏆  COIN LEADERBOARD (15‑day pnl)\n" + table.to_string(index=False, float_format="{:+.1%}".format))

#     contenders = table.head(TOP_N_LIVE)
#     if contenders.empty:
#         log.warning("No contenders found – aborting playoffs")
#         return

#     log.info("Staging playoff among top %d coins…", len(contenders))
#     for coin in contenders["coin"]:
#         run_tournament(coin, keep_top=1)  # sanity check champion still best
#     # TODO: pull ledger, lock capital, and kick off live deployment

# # ----------------------------------------------------------------------
# # Main orchestration
# # ----------------------------------------------------------------------

# def _single_pass() -> None:
#     coins_cfg = _load_coins()
#     coins = list(coins_cfg.values())
#     if not coins:
#         log.error("No coins in configuration – exiting")
#         sys.exit(1)

#     dm = DataManager(coins, data_directory=DATA_DIR, verbose=False)
#     dm.ensure_datasets()

#     rows: List[dict] = []
#     with ThreadPoolExecutor(max_workers=THREADS) as pool:
#         futs = {pool.submit(_process_coin, cfg, dm): cfg["symbol"] for cfg in coins}
#         for fut in as_completed(futs):
#             sym = futs[fut]
#             try:
#                 rows.append(fut.result())
#             except Exception as exc:
#                 log.exception("[%s] pipeline failed: %s", sym, exc)

#     dm.stop()

#     if not rows:
#         log.error("Nothing processed – abort")
#         return

#     summary = pd.DataFrame(rows)
#     _global_playoffs(summary)

# # ----------------------------------------------------------------------
# # CLI wrapper
# # ----------------------------------------------------------------------

# def main() -> None:
#     try:
#         _single_pass()
#     except KeyboardInterrupt:
#         print("\nInterrupted – goodbye!")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
readiness.py — end-to--end readiness loop (replacement)
======================================================

Replaces the old readiness_check.py.  Adds:

  • Integration with utils.utils.proxies so dashboard can reflect readiness state.
  • Per-coin failure tracking (failed_coins.json) for remediation/retry.
  • Explicit “last_action” updates: gather_data, train, tourney, backtest, size_search, ready.
  • Writes per‐coin size_configs to size_configs/<symbol>_best.json.
  • Prints a 15-day PnL leaderboard and runs a final tournament among top N champions.

Usage (from project root):
    python readiness.py

This is a one-shot run; for coins that fail, their symbols get appended to
`failed_coins.json` so you can retry them later manually or via a separate remediation script.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from utils.data_manager import DataManager
from utils.model_factory import get_champion, list_models, train_and_select
from pipelines.tournament import run_tournament
from pipelines.size_search import search_sizes
from utils.utils import configure_logging, get_logger, proxies, stop_flag

# ────────────────────────── static paths ──────────────────────────────────── #
ROOT       = Path(__file__).resolve().parents[1]          # StockFactory2/
CONFIG_YAML = ROOT / "config" / "coins.yaml"
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
SIZE_DIR    = ROOT / "size_configs"
LOGS_DIR    = ROOT / "logs"
FAILED_FILE = ROOT / "failed_coins.json"

# ────────────────────────── parameters ────────────────────────────────────── #
MIN_MODELS     = 4    # champion + 3 challengers
BACKTEST_DAYS  = 15   # look-back window for quick readiness score
TOP_N_LIVE     = 5    # number of top coins to sanity-check with a final tournament
THREADS        = min(os.cpu_count() or 8, 16)

# Ensure the necessary directories exist
SIZE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────── #
# Logging & Proxies Setup
# ──────────────────────────────────────────────────────────────────────────── #
configure_logging(LOGS_DIR)
log = get_logger("Readiness")

# Ensure proxies keys exist so dashboard won't KeyError
proxies.setdefault("last_action", "idle")
proxies.setdefault("train_jobs", 0)
proxies.setdefault("tourney_runs", 0)
proxies.setdefault("failed_jobs", 0)

# ──────────────────────────────────────────────────────────────────────────── #
# Helpers: Load coins.yaml
# ──────────────────────────────────────────────────────────────────────────── #
def _load_coins(path: Path = CONFIG_YAML) -> Dict[str, dict]:
    """
    Parse coins.yaml into a dict mapping symbol → coin config dict.
    """
    if not path.exists():
        raise FileNotFoundError(f"coins.yaml missing at {path}")
    data = yaml.safe_load(path.read_text())
    coins = data.get("coins", data) if isinstance(data, dict) else data
    if not isinstance(coins, list):
        raise ValueError("`coins.yaml` expects a list or a top-level `coins:` key")
    return {c["symbol"]: c for c in coins}


# ──────────────────────────────────────────────────────────────────────────── #
# Per-symbol model helper
# ──────────────────────────────────────────────────────────────────────────── #
def _ensure_models(cfg: dict, dm: DataManager) -> None:
    """
    Block until `cfg["symbol"]` has at least MIN_MODELS models on disk.
    1. Ensure dataset exists; else raise.
    2. Loop: train and select until #models ≥ MIN_MODELS.
    3. Final prune: run 7-day tournament to keep exactly MIN_MODELS.
    """
    symbol   = cfg["symbol"]

    # Update proxies so dashboard knows we are “gather_data”
    proxies["last_action"] = "gather_data"
    csv_path = dm._csv_path(symbol)
    if not csv_path.exists():
        csv_path = dm._latest_csv(symbol)
    if not csv_path or not csv_path.exists():
        raise RuntimeError(f"Dataset for {symbol} missing — ensure DataManager.ensure_datasets() ran successfully")

    # Train until we have MIN_MODELS
    while not stop_flag.is_set() and len(list_models(symbol)) < MIN_MODELS:
        log.info("[%s] ⏳ training (need %d models)…", symbol, MIN_MODELS)
        proxies["last_action"] = "train"
        proxies["train_jobs"] += 1
        train_and_select(cfg, csv_path)

    # Once we have ≥ MIN_MODELS, prune back to exactly MIN_MODELS via a 7-day tournament
    log.info("[%s] 🎯 pruning to exactly %d models via tournament…", symbol, MIN_MODELS)
    proxies["last_action"]  = "tourney"
    proxies["tourney_runs"] += 1
    run_tournament(symbol, keep_top=MIN_MODELS)


# ──────────────────────────────────────────────────────────────────────────── #
# Per-coin pipeline
# ──────────────────────────────────────────────────────────────────────────── #
def _process_coin(cfg: dict, dm: DataManager) -> dict:
    """
    1. Ensure models exist (≥ MIN_MODELS, pruned to exactly MIN_MODELS).
    2. Load champion; fail if none.
    3. Backtest champion over last BACKTEST_DAYS to get readiness score.
    4. Run size_config search and write JSON to size_configs/<symbol>_best.json.
    5. Return summary dict for this coin.
    """
    symbol = cfg["symbol"]
    try:
        # ① Ensure ≥ MIN_MODELS exist, pruned down to MIN_MODELS
        _ensure_models(cfg, dm)

        # ② Find champion
        champ = get_champion(symbol)
        if champ is None:
            raise RuntimeError(f"Champion not found for {symbol}")

        # ③ Backtest last BACKTEST_DAYS
        proxies["last_action"] = "backtest"
        csv_path = dm._latest_csv(symbol)
        df_all   = pd.read_csv(csv_path)
        cutoff   = pd.Timestamp.utcnow() - pd.Timedelta(days=BACKTEST_DAYS)
        df_eval  = df_all[pd.to_datetime(df_all["time"], utc=True) >= cutoff]
        if df_eval.empty:
            raise RuntimeError(f"No recent data (last {BACKTEST_DAYS} days) for {symbol}")

        # ④ search_sizes → size config + performance metrics
        proxies["last_action"] = "size_search"
        best_cfg = search_sizes(symbol, champ, cfg, df_eval)
        (SIZE_DIR / f"{symbol}_best.json").write_text(json.dumps(best_cfg, indent=2))

        # ⑤ Quick metrics: try “return” from best_cfg, else fallback to utils.quick_score
        pnl = best_cfg.get("return", float("nan"))
        if pd.isna(pnl):
            from utils.utils import quick_score
            pnl = quick_score(symbol, champ) or 0.0

        # Return summary for this coin
        return {
            "coin":     symbol,
            "models":   len(list_models(symbol)),
            "champion": champ.name,
            "return":   pnl,
        }

    except Exception:
        # Re-raise so outer loop can catch and record failures
        raise


# ──────────────────────────────────────────────────────────────────────────── #
# After all coins are done: print leaderboard & final playoff
# ──────────────────────────────────────────────────────────────────────────── #
def _global_playoffs(summary: pd.DataFrame) -> None:
    """
    1. Print a 15-day PnL leaderboard sorted descending.
    2. Pick top TOP_N_LIVE, re-run tournament on each to sanity-check champion.
    """
    table = summary.sort_values("return", ascending=False).reset_index(drop=True)
    print("\n🏆  COIN LEADERBOARD (15-day pnl)")
    print(table.to_string(index=False, float_format="{:+.2%}".format))

    contenders = table.head(TOP_N_LIVE)
    if contenders.empty:
        log.warning("No contenders found — skipping final playoffs")
        return

    log.info("🏁 Running final playoff among top %d coins…", len(contenders))
    proxies["last_action"] = "tourney"
    for coin in contenders["coin"]:
        run_tournament(coin, keep_top=1)


# ──────────────────────────────────────────────────────────────────────────── #
# Main orchestration
# ──────────────────────────────────────────────────────────────────────────── #
def _single_pass() -> None:
    # 0. Load coins
    coins_cfg = _load_coins()
    coins     = list(coins_cfg.values())
    if not coins:
        log.error("No coins in configuration — exiting")
        sys.exit(1)

    # 1. DataManager: ensure datasets (backfill if missing)
    dm = DataManager(coins, data_directory=DATA_DIR, verbose=False)
    log.info("🔄 Ensuring all datasets exist (backfill if needed)…")
    proxies["last_action"] = "gather_data"
    dm.ensure_datasets()

    # 2. Process each coin in parallel, collecting successes and failures
    rows         : List[dict] = []
    failed_list  : List[str]  = []
    proxies["failed_jobs"] = 0

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futs = {pool.submit(_process_coin, cfg, dm): cfg["symbol"] for cfg in coins}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                res = fut.result()
                rows.append(res)
                log.info("[%s] ✅ processed successfully (return %+.2f)", sym, res["return"])
            except Exception as exc:
                log.exception("[%s] pipeline failed: %s", sym, exc)
                failed_list.append(sym)
                proxies["failed_jobs"] += 1

    dm.stop()

    # 3. If everything failed, abort
    if not rows:
        log.error("🚫 Nothing processed successfully — aborting")
        sys.exit(1)

    # 4. Write failed_list to failed_coins.json so we can remediate later
    if failed_list:
        log.warning("⚠️  Some coins failed; writing %s", FAILED_FILE)
        with open(FAILED_FILE, "w", encoding="utf-8") as fh:
            json.dump({"failed": failed_list}, fh, indent=2)

    # 5. Summarize successes, run cross-coin playoffs
    summary = pd.DataFrame(rows)
    _global_playoffs(summary)

    # 6. Mark readiness done
    proxies["last_action"] = "ready"
    log.info("✅ Readiness pass complete — all done.")


# ──────────────────────────────────────────────────────────────────────────── #
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────── #
def main() -> None:
    try:
        _single_pass()
    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")
    except Exception as e:
        log.exception("🔴 Unhandled exception in readiness.py: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
