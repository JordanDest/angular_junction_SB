#!/usr/bin/env python3
"""
modelsaving_test.py – automatic dataset builder (via Kraken /Trades) and model trainer
===================================================================================

This utility script ensures that an up‑to‑date CSV OHLC dataset exists for *every*
coin listed in *config/coins.yaml* **using the /Trades endpoint**.  If a suitable
file is missing, it back‑fills the last *N* days into a fresh dataset.

After each dataset is available, the script trains one or more model variants via
:pyfunc:`utils.model_factory.train_and_select`.  Models are written to *models/*
and the best‑scoring variant is marked as "champion" (see
:pyfunc:`utils.model_factory.get_champion`).

The program is single‑threaded for clarity; feel free to wrap the inner loops in
``ThreadPoolExecutor`` if you require concurrency.

Usage (from project root)::

    python build_and_train.py --days 45 --models 3 --verbose

Arguments
---------
--coins-yaml  Path to *coins.yaml* (default: *config/coins.yaml*)
--data-dir    Directory to save CSV datasets (default: *data*)
--days        Look‑back window in days when back‑filling (default: 55)
--models      How many models to **keep** per coin (default: 2)
--verbose     Enable tqdm progress bars during back‑fill
--fast-train  Shortcut that sets the ``FAST_TRAIN`` env var, causing
              *train_and_select* to explore only the first hyper‑param combo –
              handy for CI smoke tests.

The script exits with a non‑zero code if *any* coin fails to build or train.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# ---------------------------------------------------------------------------
# Project imports – ensure repository root is on sys.path first
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from utils.data_manager import DataManager  # type: ignore
from utils.model_factory import train_grid, ensure_champion  # type: ignore

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_PATH = PROJECT_ROOT / "build_and_train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf‑8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
log = logging.getLogger("build_and_train")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_coins(cfg_path: Path) -> List[Dict]:
    """Parse *coins.yaml* into a list of dicts; fail loudly on schema errors."""
    if not cfg_path.exists():
        raise FileNotFoundError(f"Coins configuration missing: {cfg_path}")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    if not cfg or "coins" not in cfg:
        raise ValueError("'coins.yaml' must contain a top‑level 'coins:' list")
    coins: List[Dict] = cfg["coins"]
    if not coins:
        raise ValueError("'coins.yaml' lists zero coins – nothing to do")
    for c in coins:
        if "symbol" not in c:
            raise ValueError("Every coin entry needs a 'symbol' key")
    return coins

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Ensuring the datasets arent shit /Trades friendly") 
    ap.add_argument("--coins-yaml", type=Path, default=PROJECT_ROOT / "config" / "coins.yaml",
                    help="Path to coins.yaml (default: %(default)s)")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data",
                    help="Directory to store CSV datasets (default: %(default)s)")
    ap.add_argument("--days", type=int, default=55, help="Back‑fill window in days (default: 55)")
    ap.add_argument("--models", type=int, default=2, help="Models to keep per coin (default: 2)")
    ap.add_argument("--verbose", action="store_true", help="Show tqdm progress bars during back‑fill")
    ap.add_argument("--fast-train", action="store_true", help="Shortcut – sets FAST_TRAIN=1 for quicker runs")

    args = ap.parse_args()

    # Shortcut for CI: environment flag propagated into model_factory
    if args.fast_train:
        os.environ["FAST_TRAIN"] = "1"

    coins = load_coins(args.coins_yaml)
    log.info("Loaded %d coins from %s", len(coins), args.coins_yaml)

    # The DataManager handles *all* interaction with Kraken /Trades as well as
    # validating and patching any gaps.
    dm = DataManager(
        coins,
        data_directory=args.data_dir,
        default_days_back=args.days,
        verbose=args.verbose,
    )

    log.info("Ensuring datasets via /Trades … this may take a while on first run")
    dm.ensure_datasets()

    overall_failures: Dict[str, str] = {}

    for cfg in coins:
        sym = cfg["symbol"]
        csv_path = dm.latest_csv(sym)
        if csv_path is None or not csv_path.exists():
            msg = f"{sym}: dataset missing even after ensure_datasets()"
            log.error(msg)
            overall_failures[sym] = msg
            continue

        log.info("⚙️  %s – ensuring champion from %s", sym, csv_path.name)
        try:
            champ = ensure_champion(cfg, csv_path, depth="partial")
            log.info("🏅 %s – champion: %s", sym, champ.name if champ else "none")
        except Exception as exc:
            log.exception("%s – training failed: %s", sym, exc)
            overall_failures[sym] = str(exc)

    dm.stop()
# quick Example
# from utils.model_factory import ensure_champion, train_grid
# # quick smoke‑test:
# train_grid(cfg, bars=[1,3], depth="quick")
# # nightly training blast:
# train_grid(cfg, bars=[1,3,5,15,30], depth="full", preset="swing")

# # guarantee champion for profit_test:
# champ = ensure_champion(cfg, csv_path, depth="partial")
    # ------------------------------------------------------------------
    # Outcome summary & exit code
    # ------------------------------------------------------------------
    if overall_failures:
        log.error("%d/%d coins failed – see log for details", len(overall_failures), len(coins))
        for s, e in overall_failures.items():
            log.error("%s: %s", s, e)
        sys.exit(1)

    print("\n✅ All datasets built and models trained successfully!")
    print(f"Detailed logs written to {LOG_PATH}")


if __name__ == "__main__":
    main()
