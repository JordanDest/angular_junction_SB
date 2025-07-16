#!/usr/bin/env python3
"""
backfill.py – Robust Kraken /Trades backfiller (gap‐fill + sanitation).

Usage:
    python backfill.py \
        [--days-back <N>] \
        [--interval-minutes <M>] \
        [--coins <path/to/coins.yaml>] \
        [--data-dir <path/to/data/>] \
        [--verbose]

Example:
    python backfill.py \
        --days-back 30 \
        --interval-minutes 5 \
        --coins config/coins.yaml \
        --data-dir data/ \
        --verbose
"""
import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Sequence

import yaml

# We assume data_manager.py is alongside backfill.py (or in PYTHONPATH)
from data_manager import DataManager, CoinCfg

_LOG = logging.getLogger("backfill")


def load_coins_yaml(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read `coins.yaml`, which may be:
      - A mapping with "coins": [ ... ]
      - A simple list of strings or dicts

    Returns a dict: token -> coin_config (each is a dict with at least "symbol").
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – create it or pass --coins")

    raw = yaml.safe_load(path.read_text()) or {}
    if isinstance(raw, dict):
        coins_raw: Sequence = raw.get("coins", [])
    else:
        coins_raw = raw

    tokens_cfg: Dict[str, Dict[str, Any]] = {}
    # Normalize each entry
    for entry in coins_raw:
        if isinstance(entry, str):
            cfg = {"symbol": entry.upper()}
        elif isinstance(entry, dict):
            cfg = entry.copy()
            # allow "kraken_pair" as alias for "symbol"
            if "symbol" not in cfg and "kraken_pair" in cfg:
                cfg["symbol"] = cfg.pop("kraken_pair")
            cfg["symbol"] = cfg["symbol"].upper()
        else:
            _LOG.warning("Skipping invalid coin entry in YAML: %s", entry)
            continue

        # Deduce a token (strip leading X if present, like XBT→BTC, XDG→DOGE)
        sym = cfg["symbol"].upper()
        if sym.startswith("XBT"):
            token = "BTC"
        elif sym.startswith("XDG"):
            token = "DOGE"
        else:
            token = sym.lstrip("X")[:3]

        tokens_cfg[token] = cfg

    return tokens_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Kraken /Trades into OHLC CSVs (gap-fill + validation)."
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=55,
        help="How many days of history to backfill (default: 55).",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=1,
        help="Bar size in minutes (default: 1).",
    )
    parser.add_argument(
        "--coins",
        type=Path,
        default=Path("config/coins.yaml"),
        help="Path to coins.yaml file (default: config/coins.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where CSVs will be written (default: data/).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose progress bars and logging.",
    )
    args = parser.parse_args()

    # Set up basic logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # 1. Load + normalize coin configs
    try:
        tokens_cfg = load_coins_yaml(args.coins)
    except Exception as e:
        _LOG.error("Failed to load coins YAML: %s", e)
        sys.exit(1)

    # 2. Inject/override days_back and interval_minutes into each coin's config
    for token, cfg in tokens_cfg.items():
        cfg["days_back"] = args.days_back
        cfg["interval_minutes"] = args.interval_minutes

    # 3. Instantiate DataManager (which owns the backfill logic)
    #    We pass our coin_settings list, plus override data_directory.
    manager = DataManager(
        coin_settings=list(tokens_cfg.values()),
        data_directory=args.data_dir,
        default_interval_minutes=args.interval_minutes,
        default_days_back=args.days_back,
        verbose=args.verbose,
        interactive=False,  # non-interactive mode
    )

    # Ensure data directory exists
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # 4. For each coin, force _perform_backfill regardless of existing files.
    #    DataManager._perform_backfill(...) will handle streaming, aggregation, gap-fill and validation.
    for cfg in manager.coin_settings:
        symbol = cfg["symbol"]
        _LOG.info("Starting backfill for %s: %dd of %d-minute bars…",
                  symbol, cfg["days_back"], cfg["interval_minutes"])
        try:
            manager._perform_backfill(cfg)
        except Exception as e:
            _LOG.error("%s: backfill failed: %s", symbol, e)
        else:
            _LOG.info("%s: backfill complete", symbol)


if __name__ == "__main__":
    main()
