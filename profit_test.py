# # # # #!/usr/bin/env python3
# # # # """profit_test.py — End-to-end model-training + multi-window back-testing
# # # # ========================================================================= 

# # # # This script loads your coin configuration, ensures data and models exist,
# # # # and then runs backtests over multiple look-back windows to measure profitability.

# # # # Key enhancements:
# # # # ---------------
# # # # * Robust error handling: individual coin failures do not crash the entire run.
# # # # * Improved logging: concise yet descriptive messages at each stage.
# # # # * Dynamic window handling: uses actual day counts for backtesting, not indices.
# # # # * Progress bars: show per-coin and per-window progress so you know the script isn’t frozen.
# # # # * Single-pass model training: only retrains when models are missing or stale.
# # # # * Output: detailed & summary CSVs are written into `data/`, plus a JSON log of best‐performers.

# # # # Usage:
# # # # -----
# # # #     python profit_test.py                       # default 2 & 15 day windows
# # # #     python profit_test.py --windows 5 30 90     # custom windows
# # # #     python profit_test.py --no-sizes            # ignore size_configs
# # # #     python profit_test.py --no-train            # skip auto-training missing models

# # # # Example:
# # # # -----
# # # #     $ python profit_test.py
# # # #     12:00:00 |     INFO | profit_test | Requested windows: [2, 15]
# # # #     12:00:00 |     INFO | profit_test | Loaded 10 coins from config
# # # #     12:00:01 |     INFO | profit_test | Applying size overrides (if any)…
# # # #     12:00:02 |     INFO | profit_test | Ensuring data is fresh for max window=15 days…
# # # #     12:00:10 |     INFO | profit_test | Checking models (may trigger training)…
# # # #     12:05:30 |     INFO | profit_test | Models ready; discovering available models…
# # # #     12:05:30 |     INFO | profit_test | Beginning backtests over 2d window…
# # # #       2d coins:  50%|█████     | 5/10 [00:10<00:10,  2.00s/coin]
# # # #       ▷ Running models for BTC:  25%|███       | 1/4 [00:02<00:06,  2.00s/model]
# # # #       ▷ Running models for ETH:  50%|█████     | 2/4 [00:04<00:04,  2.00s/model]
# # # #       …
# # # #     12:10:00 |     INFO | profit_test | Completed 2d window backtest: saved detail to backtest_detail_20250605_121000.csv
# # # #     12:10:00 |     INFO | profit_test | Beginning backtests over 15d window…
# # # #       15d coins:  50%|█████     | 5/10 [00:12<00:12,  2.40s/coin]
# # # #       ▷ Running models for BTC:  50%|█████     | 2/4 [00:05<00:05,  2.50s/model]
# # # #       …
# # # #     12:20:00 |     INFO | profit_test | Completed 15d window backtest: saved detail to backtest_detail_20250605_122000.csv
# # # #     12:20:00 |     INFO | profit_test | Cross-window summary:
# # # #       model, coin, ret_mean, dd_worst, shr_mean
# # # #       modelA.pkl, BTC, +12.3%, -3.8%, 1.25
# # # #       modelB.pkl, ETH, +10.5%, -4.2%, 1.10
# # # #       …

# # # #     12:20:00 |     INFO | profit_test | Coin performance log updated → coin_perf.json
# # # # """

# # # # from __future__ import annotations

# # # # import argparse
# # # # import json
# # # # import logging
# # # # import pathlib
# # # # import subprocess
# # # # import sys
# # # # import time
# # # # from collections import defaultdict
# # # # from datetime import datetime, timezone, timedelta
# # # # from typing import Dict, List, Sequence
# # # # import re
# # # # from pathlib import Path
# # # # from tqdm.auto import tqdm
# # # # import pandas as pd
# # # # import numpy as np
# # # # import yaml
# # # # from tqdm.auto import tqdm, trange



# # # # from core.ktrader import KTrader
# # # # # -------------------------------------------------------------------------
# # # # # Paths & constants
# # # # # -------------------------------------------------------------------------
# # # # ROOT = pathlib.Path(__file__).resolve().parent
# # # # MODELS_DIR = ROOT / "models"
# # # # DATA_DIR = ROOT / "data"
# # # # LOGS_DIR = ROOT / "logs"
# # # # COINS_YAML = ROOT / "config" / "coins.yaml"
# # # # SIZE_CFG_DIR = ROOT / "size_configs"

# # # # STARTING_CAPITAL = 100.0
# # # # DEFAULT_WINDOWS = (2, 15)            # default look-back windows in days
# # # # STALE_DAYS = 30                       # retrain if model is older than this
# # # # COIN_LOG = LOGS_DIR / "coin_perf.json"  # JSON file for writing best‐performer per coin
# # # # FRESH_START = False                   # Force retraining all models from scratch

# # # # # Ensure necessary directories exist
# # # # for d in (MODELS_DIR, DATA_DIR, LOGS_DIR):
# # # #     d.mkdir(parents=True, exist_ok=True)

# # # # # -------------------------------------------------------------------------
# # # # # Logging setup
# # # # # -------------------------------------------------------------------------
# # # # logging.basicConfig(
# # # #     level=logging.INFO,
# # # #     format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
# # # #     datefmt="%H:%M:%S",
# # # #     handlers=[
# # # #         logging.StreamHandler(sys.stdout),
# # # #         logging.FileHandler(LOGS_DIR / "profit_test.log", mode="a", encoding="utf-8"),
# # # #     ],
# # # # )
# # # # _LOG = logging.getLogger("profit_test")

# # # # # -------------------------------------------------------------------------
# # # # # Flexible coin config loader
# # # # # -------------------------------------------------------------------------

# # # # def _deduce_token(sym: str) -> str:
# # # #     """
# # # #     Normalize symbol to a token like 'BTC', 'ETH', etc.
# # # #     Strips leading 'X' and remaps known aliases.
# # # #     """
# # # #     sym = sym.upper()
# # # #     mapping = {"XBT": "BTC", "XDG": "DOGE"}
# # # #     for k, v in mapping.items():
# # # #         if k in sym:
# # # #             return v
# # # #     return sym.lstrip("X")[:3]


# # # # def load_coins_yaml(path: pathlib.Path) -> Dict[str, dict]:
# # # #     """
# # # #     Read `coins.yaml`, which may be:
# # # #       - A mapping with "coins": [ ... ]
# # # #       - A simple list of strings or dicts
# # # #     Returns a dict: token -> coin_config.
# # # #     """
# # # #     if not path.exists():
# # # #         raise FileNotFoundError(f"{path} not found – create it or pass --coins")

# # # #     raw = yaml.safe_load(path.read_text()) or {}
# # # #     if isinstance(raw, dict):
# # # #         coins_raw: Sequence = raw.get("coins", [])
# # # #     else:
# # # #         coins_raw = raw

# # # #     tokens: Dict[str, dict] = {}
# # # #     for entry in coins_raw:
# # # #         if isinstance(entry, str):
# # # #             cfg = {"symbol": entry.upper(), "interval_minutes": 1}
# # # #         elif isinstance(entry, dict):
# # # #             cfg = entry.copy()
# # # #             if "symbol" not in cfg and "kraken_pair" in cfg:
# # # #                 cfg["symbol"] = cfg.pop("kraken_pair")
# # # #             cfg["symbol"] = cfg["symbol"].upper()
# # # #         else:
# # # #             _LOG.warning("Skipping invalid coin entry in YAML: %s", entry)
# # # #             continue

# # # #         token = _deduce_token(cfg["symbol"])
# # # #         tokens[token] = cfg
# # # #     return tokens

# # # # # -------------------------------------------------------------------------
# # # # # Merge size config (either from `trading` or fallback to local JSON)
# # # # # -------------------------------------------------------------------------
# # # # try:
# # # #     from trading import merge_size_cfg  # type: ignore
# # # # except ImportError:
# # # #     def merge_size_cfg(symbol: str, coin_cfg: dict) -> dict:
# # # #         """
# # # #         If `size_configs/{symbol}_best.json` exists, overlay its fields onto coin_cfg.
# # # #         """
# # # #         path = SIZE_CFG_DIR / f"{symbol}_best.json"
# # # #         if not path.exists():
# # # #             return coin_cfg
# # # #         try:
# # # #             overrides = json.loads(path.read_text())
# # # #             if not isinstance(overrides, dict):
# # # #                 raise TypeError("size config must be a JSON object")
# # # #             merged = {**coin_cfg, **overrides}
# # # #             _LOG.info("Applied size overrides for %s", symbol)
# # # #             return merged
# # # #         except Exception as e:
# # # #             _LOG.warning("Failed to apply size overrides for %s: %s", symbol, e)
# # # #             return coin_cfg

# # # # # -------------------------------------------------------------------------
# # # # # Model factory imports
# # # # # -------------------------------------------------------------------------
# # # # from utils.model_factory import (
# # # #     ensure_champion,
# # # #     get_champion,
# # # #     parse_model_bar
# # # # )
# # # # # We assume that `utils/model_factory.py` contains our refactored pipeline.

# # # # # -------------------------------------------------------------------------
# # # # # CSV helper: pick the newest CSV for a symbol
# # # # # -------------------------------------------------------------------------

# # # # def latest_csv(symbol: str, bar: int) -> pathlib.Path | None:
# # # #     """
# # # #     Find the newest CSV in DATA_DIR matching '{symbol}_*_{bar}m_*.csv'.
# # # #     """
# # # #     pattern = f"{symbol}_*_{bar}m_*.csv"
# # # #     candidates = list(DATA_DIR.glob(pattern))
# # # #     return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


# # # # # -------------------------------------------------------------------------
# # # # # Ensure fresh/suitable models: train if missing or stale
# # # # # -------------------------------------------------------------------------



# # # # def parse_bar_from_name(path: Path) -> int:
# # # #     """Return bar length in *minutes* from a model filename.
# # # #     Examples
# # # #     --------
# # # #     >>> parse_bar_from_name(Path("ADAUSD_1m_thr.pkl"))
# # # #     1
# # # #     >>> parse_bar_from_name(Path("BTC_h4_xyz.pkl"))
# # # #     240
# # # #     >>> parse_bar_from_name(Path("ETH_d1_...").
# # # #     1440
# # # #     """
# # # #     stem = path.stem  # no .pkl
# # # #     # _1m  _3m  _5m  _15m …
# # # #     if m := re.search(r"_(\d+)m", stem):
# # # #         return int(m.group(1))
# # # #     if m := re.search(r"_h(\d+)", stem):
# # # #         return int(m.group(1)) * 60
# # # #     if m := re.search(r"_d(\d+)", stem):
# # # #         return int(m.group(1)) * 1440
# # # #     # default to 1‑minute
# # # #     return 1

# # # # def ensure_fresh_models(
# # # #     tokens_cfg: Dict[str, dict],
# # # #     *,
# # # #     stale_days: int = STALE_DAYS,
# # # #     no_train: bool = False,
# # # # ) -> None:
# # # #     """
# # # #     For each token in tokens_cfg:
# # # #       - If FRESH_START=True → train full grid regardless of existing models
# # # #       - Else: If no champion exists → train a new one (unless --no-train)
# # # #               If champion is older than `stale_days` → retrain
# # # #     Any failure to train a champion for a coin is logged and skipped.
# # # #     """
# # # #     cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=stale_days)
# # # #     for token, cfg in tokens_cfg.items():
# # # #         sym = cfg["symbol"]
# # # #         _LOG.info("Checking model for %s …", sym)
# # # #         bar = cfg.get("interval_minutes", 1)
# # # #         csv_path = latest_csv(sym, bar)
# # # #         if csv_path is None:
# # # #             _LOG.error("No data CSV found for %s – cannot train. Skipping.", sym)
# # # #             continue

# # # #         # If forcing full retraining of all models
# # # #         if FRESH_START:
# # # #             _LOG.info("FRESH_START enabled – forcing full retrain for %s", sym)
# # # #             try:
# # # #                 # Train full grid (depth="full") for this symbol
# # # #                 ensure_champion(cfg, csv_path, depth="full", preset="scalp")
# # # #                 _LOG.info("Full retrain complete for %s", sym)
# # # #             except RuntimeError as e:
# # # #                 _LOG.error("Failed full retrain for %s: %s – skipping coin", sym, e)
# # # #             except Exception as e:
# # # #                 _LOG.exception("Unexpected error during full retrain for %s: %s – skipping coin", sym, e)
# # # #             continue

# # # #         champ_path = get_champion(sym)

# # # #         needs_training = False
# # # #         if champ_path is None:
# # # #             _LOG.info("%s – no champion found → training required", sym)
# # # #             needs_training = True
# # # #         else:
# # # #             mtime = datetime.fromtimestamp(champ_path.stat().st_mtime, timezone.utc)
# # # #             age_days = (datetime.now(timezone.utc) - mtime).days
# # # #             if age_days >= stale_days:
# # # #                 _LOG.info("%s – champion is %dd old (>= %d) → retraining",
# # # #                           sym, age_days, stale_days)
# # # #                 needs_training = True
# # # #             else:
# # # #                 _LOG.debug("%s – champion is %dd old (< %d), skipping retrain",
# # # #                            sym, age_days, stale_days)

# # # #         if not needs_training:
# # # #             continue

# # # #         if no_train:
# # # #             raise RuntimeError(f"Model for {sym} is missing/stale and --no-train was given")

# # # #         _LOG.info("Training champion for %s …", sym)
# # # #         try:
# # # #             ensure_champion(cfg, csv_path, depth="full", preset="scalp")
# # # #             _LOG.info("Champion trained (or found) for %s successfully.", sym)
# # # #         except RuntimeError as e:
# # # #             _LOG.error("Failed to train champion for %s: %s – skipping coin", sym, e)
# # # #         except Exception as e:
# # # #             _LOG.exception("Unexpected error while training %s: %s – skipping coin", sym, e)

# # # # # -------------------------------------------------------------------------
# # # # # Ensure historical data is present; calls `trading_test.fetch_data`
# # # # # -------------------------------------------------------------------------

# # # # def ensure_backfill(tokens_cfg: Dict[str, dict], max_window: int) -> None:
# # # #     """
# # # #     Use trading_test.fetch_data to fetch any missing minute-bars for all coins.
# # # #     `max_window` is the largest day-window we'll need, so we ensure at least that many days.
# # # #     """
# # # #     try:
# # # #         import trading_test as tt  # local import
# # # #     except ImportError as e:
# # # #         raise RuntimeError("trading_test.py not importable – cannot backfill data") from e

# # # #     _LOG.info("Ensuring data backfill for last %dd …", max_window + 1)
# # # #     try:
# # # #         # trading_test.fetch_data will skip any currency pairs that are already up-to-date
# # # #         tt.fetch_data(list(tokens_cfg.values()), days=(max_window + 1), data_dir=DATA_DIR)
# # # #         _LOG.info("Data backfill complete.")
# # # #     except Exception as e:
# # # #         _LOG.exception("Error during data backfill: %s", e)
# # # #         raise

# # # # # -----------------------------------------------------------------------------
# # # # # Backtest runner: run each model on its own bar-interval
# # # # # -----------------------------------------------------------------------------

# # # # def run_backtests(
# # # #     models: Sequence[tuple[str, int, pathlib.Path]],
# # # #     tokens_cfg: Dict[str, dict],
# # # #     window_days: int,
# # # # ) -> List[dict]:
# # # #     """
# # # #     models: list of (token, bar, model_path)
# # # #     tokens_cfg: mapping token → coin config (must include ["symbol"]).
# # # #     window_days: look-back in days.

# # # #     Returns a list of result dicts.
# # # #     """
# # # #     rows: List[dict] = []

# # # #     for token, bar, model_path in tqdm(models,
# # # #                                        desc=f"{window_days}d models",
# # # #                                        unit="model"):
# # # #         sym = tokens_cfg[token]["symbol"]

# # # #         # 1) Load the exact CSV for this bar
# # # #         csv_path = latest_csv(sym, bar)
# # # #         if not csv_path:
# # # #             _LOG.warning("No %dm CSV for %s → skipping", bar, sym)
# # # #             continue

# # # #         try:
# # # #             df = pd.read_csv(csv_path, engine="python")
# # # #         except Exception as e:
# # # #             _LOG.error("Failed to read %s → skipping model %s: %s",
# # # #                        csv_path.name, model_path.name, e)
# # # #             continue

# # # #         if "time" not in df.columns:
# # # #             _LOG.warning("CSV %s has no 'time' column → skipping %s",
# # # #                          csv_path.name, model_path.name)
# # # #             continue

# # # #         # 2) Parse & filter to last N days
# # # #         try:
# # # #             df["time"] = pd.to_datetime(df["time"], utc=True)
# # # #         except Exception as e:
# # # #             _LOG.error("Failed to parse 'time' in %s: %s", csv_path.name, e)
# # # #             continue

# # # #         cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
# # # #         df_slice = df[df["time"] >= cutoff]
# # # #         if df_slice.empty:
# # # #             _LOG.warning("No %dm bars for %s in %dd → skipping %s",
# # # #                          bar, sym, window_days, model_path.name)
# # # #             continue

# # # #         records = df_slice.to_dict("records")

# # # #         # 3) Instantiate trader & walk-forward
# # # #         try:
# # # #             trader = KTrader({"symbol": sym}, model_path,
# # # #                              [STARTING_CAPITAL],
# # # #                              starting_capital=STARTING_CAPITAL)
# # # #         except Exception as e:
# # # #             _LOG.error("Failed to init trader for %s with %s: %s",
# # # #                        sym, model_path.name, e)
# # # #             continue

# # # #         equity = [STARTING_CAPITAL]
# # # #         import torch
# # # #         try:
# # # #             with torch.inference_mode():
# # # #                 for bar_row in records:
# # # #                     trader.step(bar_row)
# # # #                     equity.append(trader.capital)
# # # #         except Exception as e:
# # # #             _LOG.error("Backtest error %s with %s: %s",
# # # #                        sym, model_path.name, e)
# # # #             continue

# # # #         # 4) Compute metrics
# # # #         eq = np.asarray(equity, dtype=float)
# # # #         if eq.size < 2:
# # # #             ret = dd = shr = 0.0
# # # #         else:
# # # #             ret = float(eq[-1] / eq[0] - 1)
# # # #             rets = np.diff(eq) / eq[:-1]
# # # #             rolling_max = np.maximum.accumulate(eq)
# # # #             dd = float((eq / rolling_max - 1).min())
# # # #             shr = float((rets.mean() / (rets.std() + 1e-9)) * np.sqrt(1440))

# # # #         rows.append({
# # # #             "model":    model_path.name,
# # # #             "coin":     token,
# # # #             "bar":      bar,
# # # #             "window_d": window_days,
# # # #             "return":   ret,
# # # #             "drawdown": dd,
# # # #             "sharpe":   shr,
# # # #             "ts":       datetime.now(timezone.utc).isoformat(),
# # # #         })

# # # #     return rows
# # # # def load_csv_any(symbol: str) -> pd.DataFrame | None:
# # # #     """Load the *finest* (lowest bar) CSV we can find for *symbol*."""
# # # #     cands = list(DATA_DIR.glob(f"{symbol}_*_*.csv"))
# # # #     if not cands:
# # # #         return None
# # # #     best = min(cands, key=lambda p: parse_bar_from_name(p))
# # # #     try:
# # # #         df = pd.read_csv(best)
# # # #         df["time"] = pd.to_datetime(df["time"], utc=True)
# # # #         return df
# # # #     except Exception as e:  # noqa: BLE001
# # # #         _LOG.error("Malformed CSV %s: %s", best.name, e)
# # # #         return None
# # # # def ensure_bar(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
# # # #     """Return *df* resampled to *bar_minutes* if needed (NEW ↘)."""
# # # #     first_bar = int(np.diff(df["time"].values[:2]).astype("timedelta64[m]")[0])
# # # #     if first_bar == bar_minutes:
# # # #         return df
# # # #     df = df.set_index("time").sort_index()
# # # #     agg = {
# # # #         col: "last" if col == "close" else "sum" if col == "volume" else "first"
# # # #         for col in df.columns
# # # #         if col != "time"
# # # #     }
# # # #     return df.resample(f"{bar_minutes}T").agg(agg).dropna().reset_index()
# # # # # -------------------------------------------------------------------------
# # # # # Print a leaderboard of top returns for a given window
# # # # # -------------------------------------------------------------------------

# # # # def print_leaderboard(df: pd.DataFrame, window_d: int) -> None:
# # # #     if df.empty:
# # # #         _LOG.warning("No results for %dd window", window_d)
# # # #         return

# # # #     top10 = df.sort_values("return", ascending=False).head(10)
# # # #     print(f"\n=== Top performers — {window_d}-day window ===")
# # # #     print(
# # # #         top10[["model", "coin", "return", "drawdown", "sharpe"]]
# # # #         .to_string(index=False, formatters={
# # # #             "return": "{:+.1%}".format,
# # # #             "drawdown": "{:+.1%}".format,
# # # #             "sharpe": "{:.2f}".format,
# # # #         })
# # # #     )

# # # # # -------------------------------------------------------------------------
# # # # # Main entrypoint
# # # # # -------------------------------------------------------------------------

# # # # def main() -> None:
# # # #     parser = argparse.ArgumentParser(
# # # #         description="Train missing models and back-test them across multiple windows."
# # # #     )
# # # #     parser.add_argument(
# # # #         "--windows", nargs="*", type=int, default=DEFAULT_WINDOWS,
# # # #         help="Look-back windows in days (e.g. 2 15 30)"
# # # #     )
# # # #     parser.add_argument(
# # # #         "--no-sizes", action="store_true",
# # # #         help="Ignore local size_configs overrides"
# # # #     )
# # # #     parser.add_argument(
# # # #         "--no-train", action="store_true",
# # # #         help="Do not auto-train missing or stale models"
# # # #     )
# # # #     args = parser.parse_args()

# # # #     windows = sorted(set(args.windows))
# # # #     _LOG.info("Requested windows: %s", windows)

# # # #     # 1. Load + normalize coin configs
# # # #     try:
# # # #         tokens_cfg = load_coins_yaml(COINS_YAML)
# # # #         _LOG.info("Loaded %d coins from config", len(tokens_cfg))
# # # #     except Exception as e:
# # # #         _LOG.exception("Failed to load coins YAML: %s", e)
# # # #         sys.exit(1)

# # # #     # 2. Apply size overrides (if any)
# # # #     if not args.no_sizes:
# # # #         _LOG.info("Applying size overrides (if present)…")
# # # #         for token in list(tokens_cfg):
# # # #             sym = tokens_cfg[token]["symbol"]
# # # #             tokens_cfg[token] = merge_size_cfg(sym, tokens_cfg[token])

# # # #     # 3. Ensure data is fresh
# # # #     try:
# # # #         ensure_backfill(tokens_cfg, max_window=max(windows))
# # # #     except Exception as e:
# # # #         _LOG.exception("Data backfill error: %s", e)
# # # #         sys.exit(1)

# # # #     # 4. Ensure champion models exist & are fresh or fully retrain if FRESH_START
# # # #     _LOG.info("Checking/Training champion models …")
# # # #     try:
# # # #         ensure_fresh_models(tokens_cfg, no_train=args.no_train)
# # # #     except Exception as e:
# # # #         _LOG.exception("Fatal error while ensuring fresh models: %s", e)
# # # #         sys.exit(1)

# # # #     # 5. Discover all models across all bars, normalizing to our 3-letter tokens
# # # #     all_model_paths = sorted(MODELS_DIR.glob("*.pkl"))
# # # #     models: List[tuple[str,int,pathlib.Path]] = []
# # # #     for p in all_model_paths:
# # # #         if p.name.endswith("_sc.pkl"):
# # # #             continue
# # # #         # take everything before the first "_" as the raw symbol
# # # #         raw_sym = p.stem.split("_", 1)[0].upper()
# # # #         # normalize to the 3-letter token (same as load_coins_yaml did)
# # # #         token = _deduce_token(raw_sym)
# # # #         if token not in tokens_cfg:
# # # #             _LOG.warning("Skipping model %s: no config for token %s", p.name, token)
# # # #             continue
# # # #         bar = parse_model_bar(p)
# # # #         models.append((token, bar, p))
# # # #     _LOG.info("Discovered %d models across all bars", len(models))

# # # #     # 6. Run backtests for each window (model-bar aware)
# # # #     all_results: List[dict] = []
# # # #     for wd in windows:
# # # #         _LOG.info("Starting backtests for %dd window …", wd)
# # # #         try:
# # # #             rows = run_backtests(models, tokens_cfg, wd)
# # # #         except Exception as e:
# # # #             _LOG.exception("Backtest loop failed for window %dd: %s", wd, e)
# # # #             continue

# # # #         if not rows:
# # # #             _LOG.warning("No backtest results for %dd window", wd)
# # # #         else:
# # # #             df = pd.DataFrame(rows)
# # # #             print_leaderboard(df, wd)
# # # #             # Save detailed CSV for this window
# # # #             ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
# # # #             detail_path = DATA_DIR / f"backtest_detail_{wd}d_{ts_str}.csv"
# # # #             try:
# # # #                 df.to_csv(detail_path, index=False)
# # # #                 _LOG.info("Detailed results for %dd window → %s", wd, detail_path.name)
# # # #             except Exception as e:
# # # #                 _LOG.error("Failed to write detail CSV for %dd window: %s", wd, e)

# # # #         all_results.extend(rows)

# # # #     if not all_results:
# # # #         _LOG.error("No backtest results at all – exiting.")
# # # #         sys.exit(1)

# # # #     # 7. Compute and print cross-window summary
# # # #     summary_df = (
# # # #         pd.DataFrame(all_results)
# # # #         .groupby(["model", "coin"], as_index=False)
# # # #         .agg(
# # # #             ret_mean=("return", "mean"),
# # # #             dd_worst=("drawdown", "min"),
# # # #             shr_mean=("sharpe", "mean"),
# # # #         )
# # # #         .sort_values("ret_mean", ascending=False)
# # # #     )
# # # #     print("\n=== Cross-window summary (avg return) ===")
# # # #     print(
# # # #         summary_df.head(20).to_string(index=False, formatters={
# # # #             "ret_mean": "{:+.1%}".format,
# # # #             "dd_worst": "{:+.1%}".format,
# # # #             "shr_mean": "{:.2f}".format,
# # # #         })
# # # #     )
# # # #     # Write summary CSV
# # # #     ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
# # # #     summ_path = DATA_DIR / f"backtest_summary_{ts_str}.csv"
# # # #     try:
# # # #         summary_df.to_csv(summ_path, index=False)
# # # #         _LOG.info("Summary CSV → %s", summ_path.name)
# # # #     except Exception as e:
# # # #         _LOG.error("Failed to write summary CSV: %s", e)

# # # #     # 8. Compute “show-me-the-money”: best model per coin (based on 15d PnL if exists)
# # # #     try:
# # # #         detail_all = pd.DataFrame(all_results)
# # # #         # If 15-day results exist, pivot to use that; otherwise fallback to max window available per coin
# # # #         group_cols = ["coin", "model", "window_d"]
# # # #         # For each coin/model, find PnL for window_d == 15 if exists
# # # #         pivot = (
# # # #             detail_all.set_index(group_cols)["return"]
# # # #             .unstack(fill_value=np.nan)
# # # #         )
# # # #         def pick_pnl(row: pd.Series) -> float:
# # # #             if 15 in row.index and not np.isnan(row[15]):
# # # #                 return row[15]
# # # #             return row.max(skipna=True)
# # # #         pnl_series = pivot.apply(pick_pnl, axis=1)

# # # #         mm_df = pd.DataFrame({
# # # #             "coin":   [idx[0] for idx in pnl_series.index],
# # # #             "model":  [idx[1] for idx in pnl_series.index],
# # # #             "pnl_15d": pnl_series.values
# # # #         })
# # # #         # Merge Sharpe and Drawdown (averaged across all windows)
# # # #         stats = (
# # # #             detail_all.groupby(["coin", "model"], as_index=False)
# # # #             .agg(sharpe=("sharpe", "mean"), drawdown=("drawdown", "min"))
# # # #         )
# # # #         money = mm_df.merge(stats, on=["coin", "model"] )
# # # #         # Now pick best model per coin
# # # #         best_per_coin = (
# # # #             money.sort_values("pnl_15d", ascending=False)
# # # #             .groupby("coin", as_index=False)
# # # #             .first()
# # # #             .sort_values("pnl_15d", ascending=False)
# # # #         )

# # # #         print("\n=== SHOW ME THE MONEY – best model per coin (15d PnL priority) ===")
# # # #         print(
# # # #             best_per_coin.to_string(index=False, formatters={
# # # #                 "pnl_15d":   "{:+.1%}".format,
# # # #                 "drawdown": "{:+.1%}".format,
# # # #                 "sharpe":   "{:.2f}".format,
# # # #             })
# # # #         )

# # # #         # Persist to coin_perf.json
# # # #         coin_log: dict = json.loads(COIN_LOG.read_text()) if COIN_LOG.exists() else {}
# # # #         for _, row in best_per_coin.iterrows():
# # # #             coin_log[row["coin"]] = {
# # # #                 "model": row["model"],
# # # #                 "pnl_15d": float(row["pnl_15d"]),
# # # #                 "sharpe": float(row["sharpe"]),
# # # #                 "drawdown": float(row["drawdown"]),
# # # #                 "ts": ts_str,
# # # #             }
# # # #         COIN_LOG.write_text(json.dumps(coin_log, indent=2))
# # # #         _LOG.info("Coin performance log updated → %s", COIN_LOG.name)
# # # #     except Exception as e:
# # # #         _LOG.exception("Failed to compute or write best‐model per coin: %s", e)

# # # # if __name__ == "__main__":
# # # #     try:
# # # #         main()
# # # #     except Exception as exc:
# # # #         _LOG.exception("Fatal error in main: %s", exc)
# # # #         sys.exit(1)
































# # # #!/usr/bin/env python3
# # # """profit_test.py — End-to-end model-training + multi-window back-testing

# # # Fix #2 2025-06-13
# # # ==================
# # # Changelog (since Fix #1)
# # # ------------------------
# # # * **Regex import** — adds the missing ``import re`` so the bar-parser works.
# # # * **Robust ``ensure_bar``** — safely handles <2 rows and mis-aligned indices.
# # # * **Window-aware skip notice** — logs a concise **INFO** line when a window
# # #   (e.g. 2 days) has no bars for a model instead of raising an error.
# # # * **Cleaner logging** — promotes frequent but harmless conditions (no data /
# # #   init-fail) to ``INFO`` while reserving ``ERROR`` for genuine faults.
# # # * **No functional signature changes** — CLI and object contracts are intact;
# # #   this is a pure maintenance drop-in replacement.

# # # This script loads your coin configuration, ensures data and models exist,
# # # and then runs backtests over multiple look-back windows to measure profitability.

# # # Key enhancements:
# # # ---------------
# # # * Robust error handling: individual coin failures do not crash the entire run.
# # # * Improved logging: concise yet descriptive messages at each stage.
# # # * Dynamic window handling: uses actual day counts for backtesting, not indices.
# # # * Progress bars: show per-coin and per-window progress so you know the script isn’t frozen.
# # # * Single-pass model training: only retrains when models are missing or stale.
# # # * Output: detailed & summary CSVs are written into `data/`, plus a JSON log of best‑performers.

# # # Usage:
# # # -----
# # #     python profit_test.py                       # default 2 & 15 day windows
# # #     python profit_test.py --windows 5 30 90     # custom windows
# # #     python profit_test.py --no-sizes            # ignore size_configs
# # #     python profit_test.py --no-train            # skip auto-training missing models

# # # Example:
# # # -----
# # #     $ python profit_test.py
# # #     12:00:00 |     INFO | profit_test | Requested windows: [2, 15]
# # #     12:00:00 |     INFO | profit_test | Loaded 10 coins from config
# # #     12:00:01 |     INFO | profit_test | Applying size overrides (if any)…
# # #     12:00:02 |     INFO | profit_test | Ensuring data is fresh for max window=15 days…
# # #     12:00:10 |     INFO | profit_test | Checking models (may trigger training)…
# # #     12:05:30 |     INFO | profit_test | Models ready; discovering available models…
# # #     12:05:30 |     INFO | profit_test | Beginning backtests over 2d window…
# # #       2d coins:  50%|█████     | 5/10 [00:10<00:10,  2.00s/coin]
# # #       ▷ Running models for BTC:  25%|███       | 1/4 [00:02<00:06,  2.00s/model]
# # #       ▷ Running models for ETH:  50%|█████     | 2/4 [00:04<00:04,  2.00s/model]
# # #       …
# # #     12:10:00 |     INFO | profit_test | Completed 2d window backtest: saved detail to backtest_detail_20250605_121000.csv
# # #     12:10:00 |     INFO | profit_test | Beginning backtests over 15d window…
# # #       15d coins:  50%|█████     | 5/10 [00:12<00:12,  2.40s/coin]
# # #       ▷ Running models for BTC:  50%|█████     | 2/4 [00:05<00:05,  2.50s/model]
# # #       …
# # #     12:20:00 |     INFO | profit_test | Completed 15d window backtest: saved detail to backtest_detail_20250605_122000.csv
# # #     12:20:00 |     INFO | profit_test | Cross-window summary:
# # #       model, coin, ret_mean, dd_worst, shr_mean
# # #       modelA.pkl, BTC, +12.3%, -3.8%, 1.25
# # #       modelB.pkl, ETH, +10.5%, -4.2%, 1.10
# # #       …

# # #     12:20:00 |     INFO | profit_test | Coin performance log updated → coin_perf.json
# # # """
# # # from __future__ import annotations

# # # import argparse
# # # import json
# # # import logging
# # # import pathlib
# # # import subprocess
# # # import sys
# # # import time
# # # from collections import defaultdict
# # # from datetime import datetime, timezone, timedelta
# # # from pathlib import Path
# # # from typing import Dict, List, Sequence, Tuple

# # # import re  # Added for bar-parser regex
# # # import numpy as np
# # # import pandas as pd
# # # import yaml
# # # from tqdm.auto import tqdm, trange

# # # from core.ktrader import KTrader
# # # from utils.model_factory import get_champion, ensure_champion, parse_model_bar

# # # # -------------------------------------------------------------------------
# # # # Paths & constants
# # # # -------------------------------------------------------------------------
# # # ROOT = pathlib.Path(__file__).resolve().parent
# # # MODELS_DIR = ROOT / "models"
# # # DATA_DIR = ROOT / "data"
# # # LOGS_DIR = ROOT / "logs"
# # # COINS_YAML = ROOT / "config" / "coins.yaml"
# # # SIZE_CFG_DIR = ROOT / "size_configs"
# # # COIN_LOG = LOGS_DIR / "coin_perf.json"

# # # STARTING_CAPITAL = 100.0
# # # DEFAULT_WINDOWS = (2, 15)
# # # STALE_DAYS = 30
# # # FRESH_START = False

# # # # Ensure necessary directories
# # # for d in (MODELS_DIR, DATA_DIR, LOGS_DIR):
# # #     d.mkdir(parents=True, exist_ok=True)

# # # # -------------------------------------------------------------------------
# # # # Logging setup
# # # # -------------------------------------------------------------------------
# # # logging.basicConfig(
# # #     level=logging.INFO,
# # #     format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
# # #     datefmt="%H:%M:%S",
# # #     handlers=[
# # #         logging.StreamHandler(sys.stdout),
# # #         logging.FileHandler(LOGS_DIR / "profit_test.log", mode="a", encoding="utf-8"),
# # #     ],
# # # )
# # # LOG = logging.getLogger("profit_test")

# # # # -------------------------------------------------------------------------
# # # # Flexible coin config loader
# # # # -------------------------------------------------------------------------

# # # def _deduce_token(sym: str) -> str:
# # #     """
# # #     Normalize symbol to a token like 'BTC', 'ETH', etc.
# # #     Strips leading 'X' and remaps known aliases.
# # #     """
# # #     sym = sym.upper()
# # #     mapping = {"XBT": "BTC", "XDG": "DOGE"}
# # #     for k, v in mapping.items():
# # #         if k in sym:
# # #             return v
# # #     return sym.lstrip("X")[:3]


# # # def load_coins_yaml(path: pathlib.Path) -> Dict[str, dict]:
# # #     """
# # #     Read `coins.yaml`, accept mapping or list, return token->cfg.
# # #     """
# # #     if not path.exists():
# # #         raise FileNotFoundError(f"{path} not found – create it or pass --coins")

# # #     raw = yaml.safe_load(path.read_text()) or {}
# # #     coins_raw = raw.get("coins", raw) if isinstance(raw, dict) else raw

# # #     tokens: Dict[str, dict] = {}
# # #     for entry in coins_raw:
# # #         if isinstance(entry, str):
# # #             cfg = {"symbol": entry.upper(), "interval_minutes": 1}
# # #         elif isinstance(entry, dict):
# # #             cfg = entry.copy()
# # #             if "symbol" not in cfg and "kraken_pair" in cfg:
# # #                 cfg["symbol"] = cfg.pop("kraken_pair")
# # #             cfg["symbol"] = cfg["symbol"].upper()
# # #         else:
# # #             LOG.warning("Skipping invalid coin entry: %s", entry)
# # #             continue
# # #         token = _deduce_token(cfg["symbol"])
# # #         tokens[token] = cfg
# # #     return tokens

# # # # -------------------------------------------------------------------------
# # # # Merge size config
# # # # -------------------------------------------------------------------------
# # # try:
# # #     from trading import merge_size_cfg  # type: ignore
# # # except ImportError:
# # #     def merge_size_cfg(symbol: str, coin_cfg: dict) -> dict:
# # #         path = SIZE_CFG_DIR / f"{symbol}_best.json"
# # #         if not path.exists():
# # #             return coin_cfg
# # #         try:
# # #             overrides = json.loads(path.read_text())
# # #             if not isinstance(overrides, dict):
# # #                 raise TypeError("size config must be a JSON object")
# # #             merged = {**coin_cfg, **overrides}
# # #             LOG.info("Applied size overrides for %s", symbol)
# # #             return merged
# # #         except Exception as e:
# # #             LOG.warning("Failed to apply size overrides for %s: %s", symbol, e)
# # #             return coin_cfg

# # # # -------------------------------------------------------------------------
# # # # CSV helper: pick the newest CSV for a symbol
# # # # -------------------------------------------------------------------------

# # # def latest_csv(symbol: str, bar: int) -> pathlib.Path | None:
# # #     pattern = f"{symbol}_*_{bar}m_*.csv"
# # #     cands = list(DATA_DIR.glob(pattern))
# # #     return max(cands, key=lambda p: p.stat().st_mtime, default=None)

# # # # -------------------------------------------------------------------------
# # # # Ensure fresh/suitable models: train if missing or stale
# # # # -------------------------------------------------------------------------

# # # def ensure_fresh_models(
# # #     tokens_cfg: Dict[str, dict],
# # #     *, stale_days: int = STALE_DAYS, no_train: bool = False
# # # ) -> None:
# # #     cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=stale_days)
# # #     for token, cfg in tokens_cfg.items():
# # #         sym = cfg["symbol"]
# # #         LOG.info("Checking model for %s …", sym)
# # #         bar = cfg.get("interval_minutes", 1)
# # #         csv_path = latest_csv(sym, bar)
# # #         if csv_path is None:
# # #             LOG.error("No data CSV for %s – skipping training.", sym)
# # #             continue

# # #         if FRESH_START:
# # #             LOG.info("FRESH_START – full retrain for %s", sym)
# # #             try:
# # #                 ensure_champion(cfg, csv_path, depth="full", preset="scalp")
# # #                 LOG.info("Retrain complete for %s", sym)
# # #             except Exception as e:
# # #                 LOG.error("Retrain error for %s: %s", sym, e)
# # #             continue

# # #         champ = get_champion(sym)
# # #         needs = False
# # #         if champ is None:
# # #             LOG.info("%s – no champion found → will train", sym)
# # #             needs = True
# # #         else:
# # #             mtime = datetime.fromtimestamp(champ.stat().st_mtime, timezone.utc)
# # #             age = (datetime.now(timezone.utc) - mtime).days
# # #             if age >= stale_days:
# # #                 LOG.info("%s champion %dd old → retrain", sym, age)
# # #                 needs = True
# # #             else:
# # #                 LOG.debug("%s champion fresh (%dd)", sym, age)
# # #         if not needs:
# # #             continue
# # #         if no_train:
# # #             raise RuntimeError(f"Stale model for {sym} and --no-train set")
# # #         LOG.info("Training champion for %s …", sym)
# # #         try:
# # #             ensure_champion(cfg, csv_path, depth="full", preset="scalp")
# # #             LOG.info("Champion ready for %s", sym)
# # #         except Exception as e:
# # #             LOG.error("Training failed for %s: %s", sym, e)

# # # # -------------------------------------------------------------------------
# # # # Ensure historical data is present
# # # # -------------------------------------------------------------------------

# # # def ensure_backfill(tokens_cfg: Dict[str, dict], max_window: int) -> None:
# # #     try:
# # #         import trading_test as tt
# # #     except ImportError:
# # #         raise RuntimeError("trading_test not importable")
# # #     LOG.info("Backfill data for last %dd …", max_window + 1)
# # #     tt.fetch_data(list(tokens_cfg.values()), days=max_window + 1, data_dir=DATA_DIR)
# # #     LOG.info("Data backfill done.")

# # # # -------------------------------------------------------------------------
# # # # Bar resample helper
# # # # -------------------------------------------------------------------------

# # # def load_csv_any(symbol: str) -> pd.DataFrame | None:
# # #     cands = list(DATA_DIR.glob(f"{symbol}_*_*.csv"))
# # #     if not cands:
# # #         return None
# # #     best = min(cands, key=parse_model_bar)
# # #     try:
# # #         df = pd.read_csv(best)
# # #         df["time"] = pd.to_datetime(df["time"], utc=True)
# # #         return df
# # #     except Exception as e:
# # #         LOG.error("Malformed CSV %s: %s", best.name, e)
# # #         return None


# # # def ensure_bar(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
# # #     cur = None
# # #     if len(df) >= 2:
# # #         try:
# # #             cur = int(np.diff(df["time"].values[:2]).astype("timedelta64[m]")[0])
# # #         except Exception:
# # #             cur = None
# # #     if cur == bar_minutes:
# # #         return df
# # #     if cur is None:
# # #         LOG.info("Unknown cadence (<2 rows) — resampling to %dm", bar_minutes)
# # #     df = df.set_index("time").sort_index()
# # #     agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
# # #     for c in df.columns:
# # #         agg.setdefault(c, "last")
# # #     return df.resample(f"{bar_minutes}").agg(agg).dropna().reset_index()

# # # # -----------------------------------------------------------------------------
# # # # Run backtests
# # # # -----------------------------------------------------------------------------

# # # def run_backtests(
# # #     models: Sequence[Tuple[str,int,Path]], tokens_cfg: Dict[str, dict], window_days: int
# # # ) -> List[dict]:
# # #     rows: List[dict] = []
# # #     cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
# # #     for token, bar, model_path in tqdm(models, desc=f"{window_days}d models", unit="model"):
# # #         sym = tokens_cfg[token]["symbol"]
# # #         df0 = load_csv_any(sym)
# # #         if df0 is None:
# # #             LOG.info("No CSV for %s — skip", sym)
# # #             continue
# # #         try:
# # #             df = ensure_bar(df0, bar)
# # #         except Exception as e:
# # #             LOG.error("Resample error %s→%dm: %s", sym, bar, e)
# # #             continue
# # #         df = df[df["time"] >= cutoff]
# # #         if df.empty:
# # #             LOG.info("%s %dm — no bars in last %dd — skip", sym, bar, window_days)
# # #             continue
# # #         try:
# # #             trader = KTrader({"symbol": sym}, model_path, [STARTING_CAPITAL])
# # #         except Exception as e:
# # #             LOG.info("Init-fail %s:%s — skip", sym, model_path.name)
# # #             continue
# # #         cap = STARTING_CAPITAL
# # #         for _, row in df.iterrows():
# # #             try:
# # #                 trader.step(row.to_dict())
# # #                 cap = trader.capital
# # #             except Exception:
# # #                 break
# # #         ret = cap/STARTING_CAPITAL - 1.0
# # #         rows.append({"model": model_path.name, "coin": token, "ret": ret, "window_d": window_days})
# # #     return rows

# # # # -------------------------------------------------------------------------
# # # # Print leaderboard
# # # # -------------------------------------------------------------------------

# # # def print_leaderboard(df: pd.DataFrame, window_d: int) -> None:
# # #     if df.empty:
# # #         LOG.info("No results for %dd window", window_d)
# # #         return
# # #     top = df.sort_values("ret", ascending=False).head(10)
# # #     print(f"\n=== Top performers — {window_d}-day window ===")
# # #     print(top[["model","coin","ret"]].to_string(index=False, formatters={"ret":"{:+.1%}".format}))

# # # # -------------------------------------------------------------------------
# # # # Main
# # # # -------------------------------------------------------------------------

# # # def main() -> None:
# # #     parser = argparse.ArgumentParser(description="Train models & backtest across windows.")
# # #     parser.add_argument("--windows", nargs="*", type=int, default=DEFAULT_WINDOWS)
# # #     parser.add_argument("--no-sizes", action="store_true")
# # #     parser.add_argument("--no-train", action="store_true")
# # #     args = parser.parse_args()

# # #     windows = sorted(set(args.windows))
# # #     LOG.info("Requested windows: %s", windows)

# # #     # Load coins
# # #     try:
# # #         tokens_cfg = load_coins_yaml(COINS_YAML)
# # #         LOG.info("Loaded %d coins", len(tokens_cfg))
# # #     except Exception as e:
# # #         LOG.error("Coins load fail: %s", e)
# # #         sys.exit(1)

# # #     # Size overrides
# # #     if not args.no_sizes:
# # #         LOG.info("Applying size overrides…")
# # #         for tk in list(tokens_cfg):
# # #             tokens_cfg[tk] = merge_size_cfg(tokens_cfg[tk]["symbol"], tokens_cfg[tk])

# # #     # Backfill data
# # #     try:
# # #         ensure_backfill(tokens_cfg, max(windows))
# # #     except Exception as e:
# # #         LOG.error("Data backfill fail: %s", e)
# # #         sys.exit(1)

# # #     # Train champions
# # #     LOG.info("Ensuring fresh models…")
# # #     try:
# # #         ensure_fresh_models(tokens_cfg, no_train=args.no_train)
# # #     except Exception as e:
# # #         LOG.error("Model training fatal: %s", e)
# # #         sys.exit(1)

# # #     # Discover models
# # #     all_paths = sorted(MODELS_DIR.glob("*.pkl"))
# # #     models: List[Tuple[str,int,Path]] = []
# # #     for p in all_paths:
# # #         if p.name.endswith("_sc.pkl"): continue
# # #         tok = _deduce_token(p.stem.split("_",1)[0])
# # #         if tok not in tokens_cfg:
# # #             LOG.debug("Skip %s: no config", p.name)
# # #             continue
# # #         bar = parse_model_bar(p)
# # #         models.append((tok, bar, p))
# # #     LOG.info("Discovered %d models", len(models))

# # #     # Backtests
# # #     all_rows: List[dict] = []
# # #     for wd in windows:
# # #         LOG.info("Backtesting %dd window…", wd)
# # #         rows = run_backtests(models, tokens_cfg, wd)
# # #         if not rows:
# # #             LOG.info("No results for %dd", wd)
# # #         else:
# # #             df = pd.DataFrame(rows)
# # #             print_leaderboard(df, wd)
# # #             ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
# # #             path = DATA_DIR / f"backtest_detail_{wd}d_{ts}.csv"
# # #             df.to_csv(path, index=False)
# # #             LOG.info("Detail → %s", path.name)
# # #         all_rows.extend(rows)

# # #     if not all_rows:
# # #         LOG.error("No backtest results — exit")
# # #         sys.exit(1)

# # #     # Summary
# # #     summ = (pd.DataFrame(all_rows)
# # #             .groupby(["model","coin"], as_index=False)
# # #             .agg(ret_mean=("ret","mean"), dd_worst=("ret","min"))
# # #             .sort_values("ret_mean", ascending=False))
# # #     print("\n=== Cross-window summary ===")
# # #     print(summ.head(20).to_string(index=False, formatters={"ret_mean":"{:+.1%}".format}))
# # #     ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
# # #     spath = DATA_DIR / f"backtest_summary_{ts}.csv"
# # #     summ.to_csv(spath, index=False)
# # #     LOG.info("Summary → %s", spath.name)

# # #     # Show-me-the-money
# # #     try:
# # #         df_all = pd.DataFrame(all_rows)
# # #         pivot = df_all.pivot_table(index=["coin","model"], columns="window_d", values="ret")
# # #         def pick(row): return row.get(15, row.max(skipna=True))
# # #         pnl = pivot.apply(pick, axis=1)
# # #         mm = pnl.reset_index().rename(columns={0:'pnl_15d'})
# # #         stats = df_all.groupby(["coin","model"], as_index=False).agg(sharpe=('ret','mean'), drawdown=('ret','min'))
# # #         best = mm.merge(stats,on=["coin","model"]).sort_values('pnl_15d',ascending=False).groupby('coin',as_index=False).first()
# # #         print("\n=== SHOW ME THE MONEY ===")
# # #         print(best.to_string(index=False, formatters={"pnl_15d":"{:+.1%}".format}))
# # #         log = json.loads(COIN_LOG.read_text()) if COIN_LOG.exists() else {}
# # #         tstamp = ts
# # #         for _,r in best.iterrows():
# # #             log[r['coin']]={'model':r['model'],'pnl_15d':r['pnl_15d'],'sharpe':r['sharpe'],'drawdown':r['drawdown'],'ts':tstamp}
# # #         COIN_LOG.write_text(json.dumps(log,indent=2))
# # #         LOG.info("Coin performance log → %s", COIN_LOG.name)
# # #     except Exception as e:
# # #         LOG.error("Show-me-the-money fail: %s", e)

# # # if __name__ == "__main__":
# # #     try:
# # #         main()
# # #     except KeyboardInterrupt:
# # #         LOG.warning("Interrupted by user")
# # #         sys.exit(130)
# # #     except Exception as exc:
# # #         LOG.error("Fatal: %s", exc)
# # #         sys.exit(1)




# # #!/usr/bin/env python3
# # """profit_test.py — End-to-end model-training + multi-window back-testing

# # Fix #2 2025-06-13
# # ==================
# # Changelog (since Fix #1)
# # ------------------------
# # * **Regex import** — adds the missing ``import re`` so the bar-parser works.
# # * **Robust ``ensure_bar``** — safely handles <2 rows and mis-aligned indices.
# # * **Window-aware skip notice** — logs a concise **INFO** line when a window
# #   (e.g. 2 days) has no bars for a model instead of raising an error.
# # * **Cleaner logging** — promotes frequent but harmless conditions (no data /
# #   init-fail) to ``INFO`` while reserving ``ERROR`` for genuine faults.
# # * **No functional signature changes** — CLI and object contracts are intact;
# #   this is a pure maintenance drop-in replacement.

# # This script loads your coin configuration, ensures data and models exist,
# # and then runs backtests over multiple look-back windows to measure profitability.

# # Key enhancements:
# # ---------------
# # * Robust error handling: individual coin failures do not crash the entire run.
# # * Improved logging: concise yet descriptive messages at each stage.
# # * Dynamic window handling: uses actual day counts for backtesting, not indices.
# # * Progress bars: show per-coin and per-window progress so you know the script isn’t frozen.
# # * Single-pass model training: only retrains when models are missing or stale.
# # * Output: detailed & summary CSVs are written into `data/`, plus a JSON log of best‑performers.

# # Usage:
# # -----
# #     python profit_test.py                       # default 2 & 15 day windows
# #     python profit_test.py --windows 5 30 90     # custom windows
# #     python profit_test.py --no-sizes            # ignore size_configs
# #     python profit_test.py --no-train            # skip auto-training missing models

# # Example:
# # -----
# #     $ python profit_test.py
# #     12:00:00 |     INFO | profit_test | Requested windows: [2, 15]
# #     12:00:00 |     INFO | profit_test | Loaded 10 coins from config
# #     12:00:01 |     INFO | profit_test | Applying size overrides (if any)…
# #     12:00:02 |     INFO | profit_test | Ensuring data is fresh for max window=15 days…
# #     12:00:10 |     INFO | profit_test | Checking models (may trigger training)…
# #     12:05:30 |     INFO | profit_test | Models ready; discovering available models…
# #     12:05:30 |     INFO | profit_test | Beginning backtests over 2d window…
# #       2d coins:  50%|█████     | 5/10 [00:10<00:10,  2.00s/coin]
# #       ▷ Running models for BTC:  25%|███       | 1/4 [00:02<00:06,  2.00s/model]
# #       ▷ Running models for ETH:  50%|█████     | 2/4 [00:04<00:04,  2.00s/model]
# #       …
# #     12:10:00 |     INFO | profit_test | Completed 2d window backtest: saved detail to backtest_detail_20250605_121000.csv
# #     12:10:00 |     INFO | profit_test | Beginning backtests over 15d window…
# #       15d coins:  50%|█████     | 5/10 [00:12<00:12,  2.40s/coin]
# #       ▷ Running models for BTC:  50%|█████     | 2/4 [00:05<00:05,  2.50s/model]
# #       …
# #     12:20:00 |     INFO | profit_test | Completed 15d window backtest: saved detail to backtest_detail_20250605_122000.csv
# #     12:20:00 |     INFO | profit_test | Cross-window summary:
# #       model, coin, ret_mean, dd_worst, shr_mean
# #       modelA.pkl, BTC, +12.3%, -3.8%, 1.25
# #       modelB.pkl, ETH, +10.5%, -4.2%, 1.10
# #       …

# #     12:20:00 |     INFO | profit_test | Coin performance log updated → coin_perf.json
# # """
# # from __future__ import annotations

# # import argparse
# # import json
# # import logging
# # import pathlib
# # import subprocess
# # import sys
# # import time
# # from collections import defaultdict
# # from datetime import datetime, timezone, timedelta
# # from pathlib import Path
# # from typing import Dict, List, Sequence, Tuple

# # import re  # Added for bar-parser regex
# # import numpy as np
# # import pandas as pd
# # import yaml
# # from tqdm.auto import tqdm, trange

# # from core.ktrader import KTrader
# # from utils.model_factory import get_champion, ensure_champion, parse_model_bar

# # # -------------------------------------------------------------------------
# # # Paths & constants
# # # -------------------------------------------------------------------------
# # ROOT = pathlib.Path(__file__).resolve().parent
# # MODELS_DIR = ROOT / "models"
# # DATA_DIR = ROOT / "data"
# # LOGS_DIR = ROOT / "logs"
# # COINS_YAML = ROOT / "config" / "coins.yaml"
# # SIZE_CFG_DIR = ROOT / "size_configs"
# # COIN_LOG = LOGS_DIR / "coin_perf.json"

# # STARTING_CAPITAL = 100.0
# # DEFAULT_WINDOWS = (2, 15)
# # STALE_DAYS = 30
# # FRESH_START = False

# # # Ensure necessary directories
# # for d in (MODELS_DIR, DATA_DIR, LOGS_DIR):
# #     d.mkdir(parents=True, exist_ok=True)

# # # -------------------------------------------------------------------------
# # # Logging setup
# # # -------------------------------------------------------------------------
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
# #     datefmt="%H:%M:%S",
# #     handlers=[
# #         logging.StreamHandler(sys.stdout),
# #         logging.FileHandler(LOGS_DIR / "profit_test.log", mode="a", encoding="utf-8"),
# #     ],
# # )
# # LOG = logging.getLogger("profit_test")

# # # -------------------------------------------------------------------------
# # # Flexible coin config loader
# # # -------------------------------------------------------------------------

# # def _deduce_token(sym: str) -> str:
# #     """
# #     Normalize symbol to a token like 'BTC', 'ETH', etc.
# #     Strips leading 'X' and remaps known aliases.
# #     """
# #     sym = sym.upper()
# #     mapping = {"XBT": "BTC", "XDG": "DOGE"}
# #     for k, v in mapping.items():
# #         if k in sym:
# #             return v
# #     return sym.lstrip("X")[:3]


# # def load_coins_yaml(path: pathlib.Path) -> Dict[str, dict]:
# #     """
# #     Read `coins.yaml`, accept mapping or list, return token->cfg.
# #     """
# #     if not path.exists():
# #         raise FileNotFoundError(f"{path} not found – create it or pass --coins")

# #     raw = yaml.safe_load(path.read_text()) or {}
# #     coins_raw = raw.get("coins", raw) if isinstance(raw, dict) else raw

# #     tokens: Dict[str, dict] = {}
# #     for entry in coins_raw:
# #         if isinstance(entry, str):
# #             cfg = {"symbol": entry.upper(), "interval_minutes": 1}
# #         elif isinstance(entry, dict):
# #             cfg = entry.copy()
# #             if "symbol" not in cfg and "kraken_pair" in cfg:
# #                 cfg["symbol"] = cfg.pop("kraken_pair")
# #             cfg["symbol"] = cfg["symbol"].upper()
# #         else:
# #             LOG.warning("Skipping invalid coin entry: %s", entry)
# #             continue
# #         token = _deduce_token(cfg["symbol"])
# #         tokens[token] = cfg
# #     return tokens

# # # -------------------------------------------------------------------------
# # # Merge size config
# # # -------------------------------------------------------------------------
# # try:
# #     from trading import merge_size_cfg  # type: ignore
# # except ImportError:
# #     def merge_size_cfg(symbol: str, coin_cfg: dict) -> dict:
# #         path = SIZE_CFG_DIR / f"{symbol}_best.json"
# #         if not path.exists():
# #             return coin_cfg
# #         try:
# #             overrides = json.loads(path.read_text())
# #             if not isinstance(overrides, dict):
# #                 raise TypeError("size config must be a JSON object")
# #             merged = {**coin_cfg, **overrides}
# #             LOG.info("Applied size overrides for %s", symbol)
# #             return merged
# #         except Exception as e:
# #             LOG.warning("Failed to apply size overrides for %s: %s", symbol, e)
# #             return coin_cfg

# # # -------------------------------------------------------------------------
# # # CSV helper: pick the newest CSV for a symbol
# # # -------------------------------------------------------------------------

# # def latest_csv(symbol: str, bar: int) -> pathlib.Path | None:
# #     pattern = f"{symbol}_*_{bar}m_*.csv"
# #     cands = list(DATA_DIR.glob(pattern))
# #     return max(cands, key=lambda p: p.stat().st_mtime, default=None)

# # # -------------------------------------------------------------------------
# # # Ensure fresh/suitable models: train if missing or stale
# # # -------------------------------------------------------------------------

# # def ensure_fresh_models(
# #     tokens_cfg: Dict[str, dict], *, stale_days: int = STALE_DAYS, no_train: bool = False
# # ) -> None:
# #     for token, cfg in tokens_cfg.items():
# #         sym = cfg["symbol"]
# #         LOG.info("Checking model for %s …", sym)
# #         bar = cfg.get("interval_minutes", 1)
# #         csv_path = latest_csv(sym, bar)
# #         if csv_path is None:
# #             LOG.error("No data CSV for %s – skipping training.", sym)
# #             continue

# #         if FRESH_START:
# #             LOG.info("FRESH_START – full retrain for %s", sym)
# #             try:
# #                 ensure_champion(cfg, csv_path, depth="full", preset="scalp")
# #                 LOG.info("Retrain complete for %s", sym)
# #             except Exception as e:
# #                 LOG.error("Retrain error for %s: %s", sym, e)
# #             continue

# #         champ = get_champion(sym)
# #         needs = False
# #         if champ is None:
# #             LOG.info("%s – no champion found → will train", sym)
# #             needs = True
# #         else:
# #             mtime = datetime.fromtimestamp(champ.stat().st_mtime, timezone.utc)
# #             age = (datetime.now(timezone.utc) - mtime).days
# #             if age >= stale_days:
# #                 LOG.info("%s champion %dd old → retrain", sym, age)
# #                 needs = True
# #             else:
# #                 LOG.debug("%s champion fresh (%dd)", sym, age)
# #         if not needs:
# #             continue
# #         if no_train:
# #             raise RuntimeError(f"Stale model for {sym} and --no-train set")
# #         LOG.info("Training champion for %s …", sym)
# #         try:
# #             ensure_champion(cfg, csv_path, depth="full", preset="scalp")
# #             LOG.info("Champion ready for %s", sym)
# #         except Exception as e:
# #             LOG.error("Training failed for %s: %s", sym, e)

# # # -------------------------------------------------------------------------\# Ensure historical data is present
# # # -------------------------------------------------------------------------

# # def ensure_backfill(tokens_cfg: Dict[str, dict], max_window: int) -> None:
# #     try:
# #         import trading_test as tt
# #     except ImportError:
# #         raise RuntimeError("trading_test not importable")
# #     LOG.info("Backfill data for last %dd …", max_window + 1)
# #     tt.fetch_data(list(tokens_cfg.values()), days=max_window + 1, data_dir=DATA_DIR)
# #     LOG.info("Data backfill done.")

# # # -------------------------------------------------------------------------
# # # Bar resample helper
# # # -------------------------------------------------------------------------

# # def load_csv_any(symbol: str) -> pd.DataFrame | None:
# #     cands = list(DATA_DIR.glob(f"{symbol}_*_*.csv"))
# #     if not cands:
# #         return None
# #     best = min(cands, key=lambda p: parse_model_bar(p))
# #     try:
# #         df = pd.read_csv(best)
# #         df["time"] = pd.to_datetime(df["time"], utc=True)
# #         return df
# #     except Exception as e:
# #         LOG.error("Malformed CSV %s: %s", best.name, e)
# #         return None


# # def ensure_bar(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
# #     """Return *df* resampled to *bar_minutes* if needed."""
# #     cur = None
# #     if len(df) >= 2:
# #         try:
# #             cur = int(np.diff(df["time"].values[:2]).astype("timedelta64[m]")[0])
# #         except Exception:
# #             cur = None
# #     if cur == bar_minutes:
# #         return df
# #     if cur is None:
# #         LOG.info("Unknown cadence (<2 rows) — resampling to %dm", bar_minutes)
# #     df = df.set_index("time").sort_index()
# #     agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
# #     for c in df.columns:
# #         agg.setdefault(c, "last")
# #     # Use 'min' alias instead of deprecated 'T'
# #     return df.resample(f"{bar_minutes}min").agg(agg).dropna().reset_index()

# # # -----------------------------------------------------------------------------
# # # Run backtests
# # # -----------------------------------------------------------------------------

# # def run_backtests(
# #     models: Sequence[Tuple[str,int,Path]], tokens_cfg: Dict[str, dict], window_days: int
# # ) -> List[dict]:
# #     rows: List[dict] = []
# #     cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
# #     for token, bar, model_path in tqdm(models, desc=f"{window_days}d models", unit="model"):
# #         sym = tokens_cfg[token]["symbol"]
# #         df0 = load_csv_any(sym)
# #         if df0 is None:
# #             LOG.info("No CSV for %s — skip", sym)
# #             continue
# #         try:
# #             df = ensure_bar(df0, bar)
# #         except Exception as e:
# #             LOG.error("Resample error %s→%dm: %s", sym, bar, e)
# #             continue
# #         df = df[df["time"] >= cutoff]
# #         if df.empty:
# #             LOG.info("%s %dm — no bars in last %dd — skip", sym, bar, window_days)
# #             continue
# #         try:
# #             trader = KTrader({"symbol": sym}, model_path, [STARTING_CAPITAL])
# #         except Exception:
# #             LOG.info("Init-fail %s:%s — skip", sym, model_path.name)
# #             continue
# #         cap = STARTING_CAPITAL
# #         for _, row in df.iterrows():
# #             try:
# #                 trader.step(row.to_dict())
# #                 cap = trader.capital
# #             except Exception:
# #                 break
# #         ret = cap/STARTING_CAPITAL - 1.0
# #         # Compute drawdown & sharpe
# #         # (optional: reintroduce metrics if desired)
# #         rows.append({"model": model_path.name, "coin": token, "ret": ret, "window_d": window_days})
# #     return rows

# # # -------------------------------------------------------------------------
# # # Print leaderboard
# # # -------------------------------------------------------------------------

# # def print_leaderboard(df: pd.DataFrame, window_d: int) -> None:
# #     if df.empty:
# #         LOG.info("No results for %dd window", window_d)
# #         return
# #     top = df.sort_values("ret", ascending=False).head(10)
# #     print(f"\n=== Top performers — {window_d}-day window ===")
# #     print(top[["model","coin","ret"]].to_string(index=False, formatters={"ret":"{:+.1%}".format}))

# # # -------------------------------------------------------------------------
# # # Main
# # # -------------------------------------------------------------------------

# # def main() -> None:
# #     parser = argparse.ArgumentParser(description="Train models & backtest across windows.")
# #     parser.add_argument("--windows", nargs="*", type=int, default=DEFAULT_WINDOWS)
# #     parser.add_argument("--no-sizes", action="store_true")
# #     parser.add_argument("--no-train", action="store_true")
# #     args = parser.parse_args()

# #     windows = sorted(set(args.windows))
# #     LOG.info("Requested windows: %s", windows)

# #     try:
# #         tokens_cfg = load_coins_yaml(COINS_YAML)
# #         LOG.info("Loaded %d coins", len(tokens_cfg))
# #     except Exception as e:
# #         LOG.error("Coins load fail: %s", e)
# #         sys.exit(1)

# #     if not args.no-sizes:
# #         LOG.info("Applying size overrides…")
# #         for tk in list(tokens_cfg):
# #             tokens_cfg[tk] = merge_size_cfg(tokens_cfg[tk]["symbol"], tokens_cfg[tk])

# #     try:
# #         ensure_backfill(tokens_cfg, max(windows))
# #     except Exception as e:
# #         LOG.error("Data backfill fail: %s", e)
# #         sys.exit(1)

# #     LOG.info("Ensuring fresh models…")
# #     try:
# #         ensure_fresh_models(tokens_cfg, no_train=args.no_train)
# #     except Exception as e:
# #         LOG.error("Model training fatal: %s", e)
# #         sys.exit(1)

# #     all_paths = sorted(MODELS_DIR.glob("*.pkl"))
# #     models: List[Tuple[str,int,Path]] = []
# #     for p in all_paths:
# #         if p.name.endswith("_sc.pkl"): continue
# #         tok = _deduce_token(p.stem.split("_",1)[0])
# #         if tok not in tokens_cfg:
# #             LOG.debug("Skip %s: no config", p.name)
# #             continue
# #         bar = parse_model_bar(p)
# #         models.append((tok, bar, p))
# #     LOG.info("Discovered %d models", len(models))

# #     all_rows: List[dict] = []
# #     for wd in windows:
# #         LOG.info("Backtesting %dd window…", wd)
# #         rows = run_backtests(models, tokens_cfg, wd)
# #         if not rows:
# #             LOG.info("No results for %dd", wd)
# #         else:
# #             df = pd.DataFrame(rows)
# #             print_leaderboard(df, wd)
# #             ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
# #             path = DATA_DIR / f"backtest_detail_{wd}d_{ts}.csv"
# #             df.to_csv(path, index=False)
# #             LOG.info("Detail → %s", path.name)
# #         all_rows.extend(rows)

# #     if not all_rows:
# #         LOG.error("No backtest results — exit")
# #         sys.exit(1)

# #     summary = (
# #         pd.DataFrame(all_rows)
# #         .groupby(["model","coin"], as_index=False)
# #         .agg(ret_mean=("ret","mean"), dd_worst=("ret","min"))
# #         .sort_values("ret_mean", ascending=False)
# #     )
# #     print("\n=== Cross-window summary ===")
# #     print(summary.head(20).to_string(index=False, formatters={"ret_mean":"{:+.1%}".format}))
# #     ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
# #     spath = DATA_DIR / f"backtest_summary_{ts}.csv"
# #     summary.to_csv(spath, index=False)
# #     LOG.info("Summary → %s", spath.name)

# #     try:
# #         df_all = pd.DataFrame(all_rows)
# #         pivot = df_all.pivot_table(index=["coin","model"], columns="window_d", values="ret")
# #         def pick(v): return v.get(15, v.max(skipna=True))
# #         pnl = pivot.apply(pick, axis=1)
# #         mm = pnl.reset_index().rename(columns={0:'pnl_15d'})
# #         stats = df_all.groupby(["coin","model"], as_index=False).agg(sharpe=('ret','mean'), drawdown=('ret','min'))
# #         best = (
# #             mm.merge(stats, on=["coin","model"])  
# #             .sort_values('pnl_15d', ascending=False)
# #             .groupby('coin', as_index=False)
# #             .first()
# #         )
# #         print("\n=== SHOW ME THE MONEY ===")
# #         print(best.to_string(index=False, formatters={"pnl_15d":"{:+.1%}".format, "drawdown":"{:+.1%}".format, "sharpe":"{:.2f}".format}))
# #         log_data = json.loads(COIN_LOG.read_text()) if COIN_LOG.exists() else {}
# #         for _, r in best.iterrows():
# #             log_data[r['coin']] = {
# #                 'model': r['model'], 'pnl_15d': float(r['pnl_15d']),
# #                 'sharpe': float(r['sharpe']), 'drawdown': float(r['drawdown']),
# #                 'ts': ts
# #             }
# #         COIN_LOG.write_text(json.dumps(log_data, indent=2))
# #         LOG.info("Coin performance log → %s", COIN_LOG.name)
# #     except Exception as e:
# #         LOG.error("Show-me-the-money fail: %s", e)

# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except KeyboardInterrupt:
# #         LOG.warning("Interrupted by user")
# #         sys.exit(130)
# #     except Exception as exc:
# #         LOG.error("Fatal: %s", exc)
# #         sys.exit(1)




# #!/usr/bin/env python3
# """profit_test.py — End-to-end model-training + multi-window back-testing

# Fix #2 2025-06-13
# ==================
# Changelog (since Fix #1)
# ------------------------
# * **Bar CSV selector** — use `parse_model_bar` (imported) to pick the finest CSV.
# * **Robust `ensure_bar`** — safely handles <2 rows, mis-aligned indices, and uses 'min' unit.
# * **No functional signature changes** — CLI and object contracts are intact; pure maintenance.

# This script loads your coin configuration, ensures data and models exist,
# and then runs backtests over multiple look-back windows to measure profitability.

# Key enhancements:
# ---------------
# * Robust error handling: individual coin failures do not crash the entire run.
# * Improved logging: concise yet descriptive messages at each stage.
# * Dynamic window handling: uses actual day counts for backtesting, not indices.
# * Progress bars: show per-coin and per-window progress so you know the script isn’t frozen.
# * Single-pass model training: only retrains when models are missing or stale.
# * Output: detailed & summary CSVs are written into `data/`, plus a JSON log of best-performers.

# Usage:
# -----
#     python profit_test.py                       # default 2 & 15 day windows
#     python profit_test.py --windows 5 30 90     # custom windows
#     python profit_test.py --no-sizes            # ignore size_configs
#     python profit_test.py --no-train            # skip auto-training missing models
# """
# from __future__ import annotations

# import argparse
# import json
# import logging
# import pathlib
# import subprocess
# import sys
# import time
# from collections import defaultdict
# from datetime import datetime, timezone, timedelta
# from pathlib import Path
# from typing import Dict, List, Sequence, Tuple

# import re
# import numpy as np
# import pandas as pd
# import yaml
# from tqdm.auto import tqdm, trange

# from core.ktrader import KTrader
# from utils.model_factory import get_champion, ensure_champion, parse_model_bar

# # -------------------------------------------------------------------------
# # Paths & constants
# # -------------------------------------------------------------------------
# ROOT = pathlib.Path(__file__).resolve().parent
# MODELS_DIR = ROOT / "models"
# DATA_DIR = ROOT / "data"
# LOGS_DIR = ROOT / "logs"
# COINS_YAML = ROOT / "config" / "coins.yaml"
# SIZE_CFG_DIR = ROOT / "size_configs"
# COIN_LOG = LOGS_DIR / "coin_perf.json"

# STARTING_CAPITAL = 100.0
# DEFAULT_WINDOWS = (2, 15)
# STALE_DAYS = 30
# FRESH_START = False

# # Ensure necessary directories exist
# for d in (MODELS_DIR, DATA_DIR, LOGS_DIR):
#     d.mkdir(parents=True, exist_ok=True)

# # -------------------------------------------------------------------------
# # Logging setup
# # -------------------------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler(LOGS_DIR / "profit_test.log", mode="a", encoding="utf-8"),
#     ],
# )
# LOG = logging.getLogger("profit_test")

# # -------------------------------------------------------------------------
# # Flexible coin config loader
# # -------------------------------------------------------------------------
# def _deduce_token(sym: str) -> str:
#     """
#     Normalize symbol to a token like 'BTC', 'ETH', etc.
#     Strips leading 'X' and remaps known aliases.
#     """
#     sym = sym.upper()
#     mapping = {"XBT": "BTC", "XDG": "DOGE"}
#     for k, v in mapping.items():
#         if k in sym:
#             return v
#     return sym.lstrip("X")[:3]

# def load_coins_yaml(path: pathlib.Path) -> Dict[str, dict]:
#     """
#     Read `coins.yaml`, accept mapping or list, return token->cfg.
#     """
#     if not path.exists():
#         raise FileNotFoundError(f"{path} not found – create it or pass --coins")

#     raw = yaml.safe_load(path.read_text()) or {}
#     coins_raw = raw.get("coins", raw) if isinstance(raw, dict) else raw

#     tokens: Dict[str, dict] = {}
#     for entry in coins_raw:
#         if isinstance(entry, str):
#             cfg = {"symbol": entry.upper(), "interval_minutes": 1}
#         elif isinstance(entry, dict):
#             cfg = entry.copy()
#             if "symbol" not in cfg and "kraken_pair" in cfg:
#                 cfg["symbol"] = cfg.pop("kraken_pair")
#             cfg["symbol"] = cfg["symbol"].upper()
#         else:
#             LOG.warning("Skipping invalid coin entry: %s", entry)
#             continue
#         token = _deduce_token(cfg["symbol"])
#         tokens[token] = cfg
#     return tokens

# # -------------------------------------------------------------------------
# # Merge size config (either from `trading` or fallback to local JSON)
# # -------------------------------------------------------------------------
# try:
#     from trading import merge_size_cfg  # type: ignore
# except ImportError:
#     def merge_size_cfg(symbol: str, coin_cfg: dict) -> dict:
#         """
#         If `size_configs/{symbol}_best.json` exists, overlay its fields onto coin_cfg.
#         """
#         path = SIZE_CFG_DIR / f"{symbol}_best.json"
#         if not path.exists():
#             return coin_cfg
#         try:
#             overrides = json.loads(path.read_text())
#             if not isinstance(overrides, dict):
#                 raise TypeError("size config must be a JSON object")
#             merged = {**coin_cfg, **overrides}
#             LOG.info("Applied size overrides for %s", symbol)
#             return merged
#         except Exception as e:
#             LOG.warning("Failed to apply size overrides for %s: %s", symbol, e)
#             return coin_cfg

# # -------------------------------------------------------------------------
# # CSV helper: pick the newest CSV for a symbol
# # -------------------------------------------------------------------------
# def latest_csv(symbol: str, bar: int) -> pathlib.Path | None:
#     """
#     Find the newest CSV in DATA_DIR matching '{symbol}_*_{bar}m_*.csv'.
#     """
#     pattern = f"{symbol}_*_{bar}m_*.csv"
#     candidates = list(DATA_DIR.glob(pattern))
#     return max(candidates, key=lambda p: p.stat().st_mtime, default=None)

# # -------------------------------------------------------------------------
# # Ensure fresh/suitable models: train if missing or stale
# # -------------------------------------------------------------------------
# def ensure_fresh_models(
#     tokens_cfg: Dict[str, dict],
#     *, stale_days: int = STALE_DAYS, no_train: bool = False
# ) -> None:
#     """
#     For each token: train if no champion or champion older than stale_days.
#     """
#     cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=stale_days)
#     for token, cfg in tokens_cfg.items():
#         sym = cfg["symbol"]
#         LOG.info("Checking model for %s …", sym)
#         bar = cfg.get("interval_minutes", 1)
#         csv_path = latest_csv(sym, bar)
#         if csv_path is None:
#             LOG.error("No data CSV for %s – skipping training.", sym)
#             continue

#         if FRESH_START:
#             LOG.info("FRESH_START enabled – forcing full retrain for %s", sym)
#             try:
#                 ensure_champion(cfg, csv_path, depth="full", preset="scalp")
#                 LOG.info("Full retrain complete for %s", sym)
#             except Exception as e:
#                 LOG.error("Retrain error for %s: %s", sym, e)
#             continue

#         champ_path = get_champion(sym)
#         needs_training = False
#         if champ_path is None:
#             LOG.info("%s – no champion found → training required", sym)
#             needs_training = True
#         else:
#             mtime = datetime.fromtimestamp(champ_path.stat().st_mtime, timezone.utc)
#             age_days = (datetime.now(timezone.utc) - mtime).days
#             if age_days >= stale_days:
#                 LOG.info("%s – champion is %dd old (>= %d) → retraining", sym, age_days, stale_days)
#                 needs_training = True
#             else:
#                 LOG.debug("%s – champion is %dd old (< %d), skipping retrain", sym, age_days, stale_days)

#         if not needs_training:
#             continue

#         if no_train:
#             raise RuntimeError(f"Model for {sym} is missing/stale and --no-train was given")

#         LOG.info("Training champion for %s …", sym)
#         try:
#             ensure_champion(cfg, csv_path, depth="full", preset="scalp")
#             LOG.info("Champion trained for %s", sym)
#         except Exception as e:
#             LOG.error("Training failed for %s: %s", sym, e)

# # -------------------------------------------------------------------------
# # Ensure historical data is present; calls `trading_test.fetch_data`
# # -------------------------------------------------------------------------
# def ensure_backfill(tokens_cfg: Dict[str, dict], max_window: int) -> None:
#     try:
#         import trading_test as tt
#     except ImportError:
#         raise RuntimeError("trading_test.py not importable – cannot backfill data")
#     LOG.info("Ensuring data backfill for last %dd …", max_window + 1)
#     tt.fetch_data(list(tokens_cfg.values()), days=(max_window + 1), data_dir=DATA_DIR)
#     LOG.info("Data backfill complete.")

# # -----------------------------------------------------------------------------
# # Bar resample helper
# # -----------------------------------------------------------------------------
# def load_csv_any(symbol: str) -> pd.DataFrame | None:
#     """
#     Load the *finest* (lowest bar) CSV we can find for *symbol*.
#     """
#     cands = list(DATA_DIR.glob(f"{symbol}_*_*.csv"))
#     if not cands:
#         return None
#     # pick by parse_model_bar, since parse_bar_from_name isn't imported here
#     best = min(cands, key=lambda p: parse_model_bar(p))
#     try:
#         df = pd.read_csv(best)
#         df["time"] = pd.to_datetime(df["time"], utc=True)
#         return df
#     except Exception as e:
#         LOG.error("Malformed CSV %s: %s", best.name, e)
#         return None

# def ensure_bar(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
#     """
#     Return *df* resampled to *bar_minutes* if needed.
#     Safely handles <2 rows and uses 'min' unit to avoid FutureWarning.
#     """
#     cur = None
#     if len(df) >= 2:
#         try:
#             cur = int(np.diff(df["time"].values[:2]).astype("timedelta64[m]")[0])
#         except Exception:
#             cur = None
#     if cur == bar_minutes:
#         return df
#     if cur is None:
#         LOG.info("Unknown cadence (<2 rows) — resampling to %dm", bar_minutes)
#     df = df.set_index("time").sort_index()
#     agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
#     for c in df.columns:
#         agg.setdefault(c, "last")
#     # Use 'min' unit instead of deprecated 'T'
#     return df.resample(f"{bar_minutes}min").agg(agg).dropna().reset_index()

# # -----------------------------------------------------------------------------
# # Backtest runner: run each model on its own bar-interval
# # -----------------------------------------------------------------------------
# def run_backtests(
#     models: Sequence[tuple[str, int, pathlib.Path]],
#     tokens_cfg: Dict[str, dict],
#     window_days: int,
# ) -> List[dict]:
#     rows: List[dict] = []
#     cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)

#     for token, bar, model_path in tqdm(models, desc=f"{window_days}d models", unit="model"):
#         sym = tokens_cfg[token]["symbol"]
#         df0 = load_csv_any(sym)
#         if df0 is None:
#             LOG.info("No CSV for %s — skip", sym)
#             continue
#         try:
#             df = ensure_bar(df0, bar)
#         except Exception as e:
#             LOG.error("Resample error %s→%dm: %s", sym, bar, e)
#             continue
#         df = df[df["time"] >= cutoff]
#         if df.empty:
#             LOG.info("%s %dm — no bars in last %dd — skip", sym, bar, window_days)
#             continue
#         try:
#             trader = KTrader({"symbol": sym}, model_path, [STARTING_CAPITAL])
#         except Exception:
#             LOG.info("Init-fail %s:%s — skip", sym, model_path.name)
#             continue
#         cap = STARTING_CAPITAL
#         for _, row in df.iterrows():
#             try:
#                 trader.step(row.to_dict())
#                 cap = trader.capital
#             except Exception:
#                 break
#         ret = cap / STARTING_CAPITAL - 1.0
#         rows.append({"model": model_path.name, "coin": token, "ret": ret, "window_d": window_days})
#     return rows

# # -------------------------------------------------------------------------
# # Print leaderboard
# # -------------------------------------------------------------------------
# def print_leaderboard(df: pd.DataFrame, window_d: int) -> None:
#     if df.empty:
#         LOG.info("No results for %dd window", window_d)
#         return
#     top10 = df.sort_values("ret", ascending=False).head(10)
#     print(f"\n=== Top performers — {window_d}-day window ===")
#     print(
#         top10[["model", "coin", "ret"]]
#         .to_string(index=False, formatters={"ret": "{:+.1%}".format})
#     )

# # -------------------------------------------------------------------------
# # Main entrypoint
# # -------------------------------------------------------------------------
# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Train missing models and back-test them across multiple windows."
#     )
#     parser.add_argument("--windows", nargs="*", type=int, default=DEFAULT_WINDOWS,
#                         help="Look-back windows in days (e.g. 2 15 30)")
#     parser.add_argument("--no-sizes", action="store_true",
#                         help="Ignore local size_configs overrides")
#     parser.add_argument("--no-train", action="store_true",
#                         help="Do not auto-train missing or stale models")
#     args = parser.parse_args()

#     windows = sorted(set(args.windows))
#     LOG.info("Requested windows: %s", windows)

#     # 1. Load + normalize coin configs
#     try:
#         tokens_cfg = load_coins_yaml(COINS_YAML)
#         LOG.info("Loaded %d coins from config", len(tokens_cfg))
#     except Exception as e:
#         LOG.error("Failed to load coins YAML: %s", e)
#         sys.exit(1)

#     # 2. Apply size overrides (if any)
#     if not args.no_sizes:
#         LOG.info("Applying size overrides (if present)…")
#         for token in list(tokens_cfg):
#             sym = tokens_cfg[token]["symbol"]
#             tokens_cfg[token] = merge_size_cfg(sym, tokens_cfg[token])

#     # 3. Ensure data is fresh
#     try:
#         ensure_backfill(tokens_cfg, max_window=max(windows))
#     except Exception as e:
#         LOG.error("Data backfill error: %s", e)
#         sys.exit(1)

#     # 4. Ensure champion models exist & are fresh
#     LOG.info("Checking/Training champion models …")
#     try:
#         ensure_fresh_models(tokens_cfg, no_train=args.no_train)
#     except Exception as e:
#         LOG.error("Fatal error while ensuring fresh models: %s", e)
#         sys.exit(1)

#     # 5. Discover all models
#     all_model_paths = sorted(MODELS_DIR.glob("*.pkl"))
#     models: List[tuple[str, int, pathlib.Path]] = []
#     for p in all_model_paths:
#         if p.name.endswith("_sc.pkl"):
#             continue
#         raw_sym = p.stem.split("_", 1)[0].upper()
#         token = _deduce_token(raw_sym)
#         if token not in tokens_cfg:
#             LOG.warning("Skipping model %s: no config for token %s", p.name, token)
#             continue
#         bar = parse_model_bar(p)
#         models.append((token, bar, p))
#     LOG.info("Discovered %d models across all bars", len(models))

#     # 6. Run backtests for each window
#     all_results: List[dict] = []
#     for wd in windows:
#         LOG.info("Starting backtests for %dd window …", wd)
#         try:
#             rows = run_backtests(models, tokens_cfg, wd)
#         except Exception as e:
#             LOG.exception("Backtest loop failed for window %dd: %s", wd, e)
#             continue

#         if not rows:
#             LOG.info("No backtest results for %dd window", wd)
#         else:
#             df = pd.DataFrame(rows)
#             print_leaderboard(df, wd)
#             ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#             detail_path = DATA_DIR / f"backtest_detail_{wd}d_{ts_str}.csv"
#             try:
#                 df.to_csv(detail_path, index=False)
#                 LOG.info("Detailed results for %dd window → %s", wd, detail_path.name)
#             except Exception as e:
#                 LOG.error("Failed to write detail CSV for %dd window: %s", wd, e)
#         all_results.extend(rows)

#     if not all_results:
#         LOG.error("No backtest results at all – exiting.")
#         sys.exit(1)

#     # 7. Compute and print cross-window summary
#     summary_df = (
#         pd.DataFrame(all_results)
#         .groupby(["model", "coin"], as_index=False)
#         .agg(ret_mean=("ret", "mean"), dd_worst=("drawdown", "min"), shr_mean=("sharpe", "mean"))
#         .sort_values("ret_mean", ascending=False)
#     )
#     print("\n=== Cross-window summary (avg return) ===")
#     print(
#         summary_df.head(20).to_string(index=False, formatters={
#             "ret_mean": "{:+.1%}".format,
#             "dd_worst": "{:+.1%}".format,
#             "shr_mean": "{:.2f}".format,
#         })
#     )
#     ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#     summ_path = DATA_DIR / f"backtest_summary_{ts_str}.csv"
#     try:
#         summary_df.to_csv(summ_path, index=False)
#         LOG.info("Summary CSV → %s", summ_path.name)
#     except Exception as e:
#         LOG.error("Failed to write summary CSV: %s", e)

#     # 8. Compute “show-me-the-money”: best model per coin
#     try:
#         detail_all = pd.DataFrame(all_results)
#         pivot = detail_all.set_index(["coin", "model", "window_d"])["ret"].unstack(fill_value=np.nan)
#         def pick_pnl(row: pd.Series) -> float:
#             if 15 in row.index and not np.isnan(row[15]):
#                 return row[15]
#             return row.max(skipna=True)
#         pnl_series = pivot.apply(pick_pnl, axis=1)
#         mm_df = pd.DataFrame({
#             "coin":    [idx[0] for idx in pnl_series.index],
#             "model":   [idx[1] for idx in pnl_series.index],
#             "pnl_15d": pnl_series.values
#         })
#         stats = (
#             detail_all.groupby(["coin", "model"], as_index=False)
#             .agg(sharpe=("sharpe", "mean"), drawdown=("drawdown", "min"))
#         )
#         best_per_coin = (
#             mm_df.merge(stats, on=["coin", "model"])
#             .sort_values("pnl_15d", ascending=False)
#             .groupby("coin", as_index=False)
#             .first()
#             .sort_values("pnl_15d", ascending=False)
#         )
#         print("\n=== SHOW ME THE MONEY – best model per coin (15d PnL priority) ===")
#         print(
#             best_per_coin.to_string(index=False, formatters={
#                 "pnl_15d":   "{:+.1%}".format,
#                 "drawdown": "{:+.1%}".format,
#                 "sharpe":   "{:.2f}".format,
#             })
#         )
#         coin_log: dict = json.loads(COIN_LOG.read_text()) if COIN_LOG.exists() else {}
#         for _, row in best_per_coin.iterrows():
#             coin_log[row["coin"]] = {
#                 "model":     row["model"],
#                 "pnl_15d":   float(row["pnl_15d"]),
#                 "sharpe":    float(row["sharpe"]),
#                 "drawdown":  float(row["drawdown"]),
#                 "ts":        ts_str,
#             }
#         COIN_LOG.write_text(json.dumps(coin_log, indent=2))
#         LOG.info("Coin performance log updated → %s", COIN_LOG.name)
#     except Exception as e:
#         LOG.exception("Failed to compute/write best-model per coin: %s", e)

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         LOG.warning("Interrupted by user")
#         sys.exit(130)
#     except Exception as exc:
#         LOG.error("Fatal: %s", exc)
#         sys.exit(1)
#!/usr/bin/env python3
"""profit_test.py — End-to-end model-training + multi-window back-testing

Fix #3 2025-06-13
==================
Changelog (since Fix #2)
------------------------
* **Quiet skips** — downgrades per-model “no CSV” / “no bars” / “init-fail” / “unknown cadence” messages from INFO → DEBUG.
* **Maintains INFO** for high-level progress, errors, and summary outputs only.
* **Resample unit** stays as `'min'` (not deprecated `'T'`).

Usage:
    python profit_test.py                       # default 2 & 15 day windows
    python profit_test.py --windows 5 30 90     # custom windows
    python profit_test.py --no-sizes            # ignore size_configs
    python profit_test.py --no-train            # skip auto-training missing models
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import re
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from core.ktrader import KTrader
from utils.model_factory import get_champion, ensure_champion, parse_model_bar

# -------------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
COINS_YAML = ROOT / "config" / "coins.yaml"
SIZE_CFG_DIR = ROOT / "size_configs"
COIN_LOG = LOGS_DIR / "coin_perf.json"

STARTING_CAPITAL = 100.0
DEFAULT_WINDOWS = (10, 45)
STALE_DAYS = 15
FRESH_START = False

# Ensure necessary directories
for d in (MODELS_DIR, DATA_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "profit_test.log", mode="a", encoding="utf-8"),
    ],
)
LOG = logging.getLogger("profit_test")

# -------------------------------------------------------------------------
# Flexible coin config loader
# -------------------------------------------------------------------------
def _deduce_token(sym: str) -> str:
    sym = sym.upper()
    mapping = {"XBT": "BTC", "XDG": "DOGE"}
    for k, v in mapping.items():
        if k in sym:
            return v
    return sym.lstrip("X")[:3]

def load_coins_yaml(path: pathlib.Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – create it or pass --coins")
    raw = yaml.safe_load(path.read_text()) or {}
    coins_raw = raw.get("coins", raw) if isinstance(raw, dict) else raw

    tokens: Dict[str, dict] = {}
    for entry in coins_raw:
        if isinstance(entry, str):
            cfg = {"symbol": entry.upper(), "interval_minutes": 1}
        elif isinstance(entry, dict):
            cfg = entry.copy()
            if "symbol" not in cfg and "kraken_pair" in cfg:
                cfg["symbol"] = cfg.pop("kraken_pair")
            cfg["symbol"] = cfg["symbol"].upper()
        else:
            LOG.warning("Skipping invalid coin entry: %s", entry)
            continue
        tokens[_deduce_token(cfg["symbol"])] = cfg
    return tokens

# -------------------------------------------------------------------------
# Merge size config
# -------------------------------------------------------------------------
try:
    from trading import merge_size_cfg  # type: ignore
except ImportError:
    def merge_size_cfg(symbol: str, coin_cfg: dict) -> dict:
        path = SIZE_CFG_DIR / f"{symbol}_best.json"
        if not path.exists():
            return coin_cfg
        try:
            overrides = json.loads(path.read_text())
            if not isinstance(overrides, dict):
                raise TypeError("size config must be a JSON object")
            merged = {**coin_cfg, **overrides}
            LOG.info("Applied size overrides for %s", symbol)
            return merged
        except Exception as e:
            LOG.warning("Failed to apply size overrides for %s: %s", symbol, e)
            return coin_cfg

# -------------------------------------------------------------------------
# CSV helper: pick the newest CSV for a symbol
# -------------------------------------------------------------------------
def latest_csv(symbol: str, bar: int) -> pathlib.Path | None:
    pattern = f"{symbol}_*_{bar}m_*.csv"
    candidates = list(DATA_DIR.glob(pattern))
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)

# -------------------------------------------------------------------------
# Ensure fresh/suitable models: train if missing or stale
# -------------------------------------------------------------------------
def ensure_fresh_models(
    tokens_cfg: Dict[str, dict],
    *, stale_days: int = STALE_DAYS, no_train: bool = False
) -> None:
    for token, cfg in tokens_cfg.items():
        sym = cfg["symbol"]
        LOG.info("Checking model for %s …", sym)
        bar = cfg.get("interval_minutes", 1)
        csv_path = latest_csv(sym, bar)
        if csv_path is None:
            LOG.error("No data CSV for %s – skipping training.", sym)
            continue

        if FRESH_START:
            LOG.info("FRESH_START enabled – full retrain for %s", sym)
            try:
                ensure_champion(cfg, csv_path, depth="full", preset="scalp")
                LOG.info("Retrain complete for %s", sym)
            except Exception as e:
                LOG.error("Retrain error for %s: %s", sym, e)
            continue

        champ_path = get_champion(sym)
        needs = False
        if champ_path is None:
            LOG.info("%s – no champion → training", sym)
            needs = True
        else:
            age = (datetime.now(timezone.utc) - datetime.fromtimestamp(champ_path.stat().st_mtime, timezone.utc)).days
            if age >= stale_days:
                LOG.info("%s champion %dd old (>= %d) → retrain", sym, age, stale_days)
                needs = True
            else:
                LOG.debug("%s champion fresh (%dd)", sym, age)
        if not needs:
            continue

        if no_train:
            raise RuntimeError(f"Stale model for {sym} and --no-train set")

        LOG.info("Training champion for %s …", sym)
        try:
            ensure_champion(cfg, csv_path, depth="full", preset="scalp")
            LOG.info("Champion ready for %s", sym)
        except Exception as e:
            LOG.error("Training failed for %s: %s", sym, e)

# -------------------------------------------------------------------------
# Ensure historical data is present
# -------------------------------------------------------------------------
def ensure_backfill(tokens_cfg: Dict[str, dict], max_window: int) -> None:
    try:
        import trading_test as tt
    except ImportError:
        raise RuntimeError("trading_test.py not importable")
    LOG.info("Backfill data for last %dd …", max_window + 1)
    tt.fetch_data(list(tokens_cfg.values()), days=max_window + 1, data_dir=DATA_DIR)
    LOG.info("Data backfill done.")

# -----------------------------------------------------------------------------
# Bar resample helper
# -----------------------------------------------------------------------------
def load_csv_any(symbol: str) -> pd.DataFrame | None:
    cands = list(DATA_DIR.glob(f"{symbol}_*_*.csv"))
    if not cands:
        return None
    best = min(cands, key=lambda p: parse_model_bar(p))
    try:
        df = pd.read_csv(best)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    except Exception as e:
        LOG.error("Malformed CSV %s: %s", best.name, e)
        return None

def ensure_bar(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    cur = None
    if len(df) >= 2:
        try:
            cur = int(np.diff(df["time"].values[:2]).astype("timedelta64[m]")[0])
        except Exception:
            cur = None
    if cur == bar_minutes:
        return df
    LOG.debug("Unknown cadence (<2 rows) or mismatch (%s vs %sm) — resampling", cur, bar_minutes)
    df = df.set_index("time").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in df.columns:
        agg.setdefault(c, "last")
    return df.resample(f"{bar_minutes}min").agg(agg).dropna().reset_index()

# -----------------------------------------------------------------------------
# Backtest runner: run each model on its own bar-interval
# -----------------------------------------------------------------------------
def run_backtests(
    models: Sequence[tuple[str, int, Path]],
    tokens_cfg: Dict[str, dict],
    window_days: int,
) -> List[dict]:
    rows: List[dict] = []
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)

    for token, bar, model_path in tqdm(models, desc=f"{window_days}d models", unit="model"):
        sym = tokens_cfg[token]["symbol"]
        df0 = load_csv_any(sym)
        if df0 is None:
            LOG.debug("No CSV for %s — skip", sym)
            continue
        try:
            df = ensure_bar(df0, bar)
        except Exception as e:
            LOG.error("Resample error %s→%dm: %s", sym, bar, e)
            continue
        df = df[df["time"] >= cutoff]
        if df.empty:
            LOG.debug("%s %dm — no bars in last %dd — skip", sym, bar, window_days)
            continue
        try:
            trader = KTrader({"symbol": sym}, model_path, [STARTING_CAPITAL])
        except Exception:
            LOG.debug("Init-fail %s:%s — skip", sym, model_path.name)
            continue
        cap = STARTING_CAPITAL
        for _, row in df.iterrows():
            try:
                trader.step(row.to_dict())
                cap = trader.capital
            except Exception:
                break
        ret = cap / STARTING_CAPITAL - 1.0
        rows.append({"model": model_path.name, "coin": token, "ret": ret, "window_d": window_days})
    return rows

# -------------------------------------------------------------------------
# Print leaderboard
# -------------------------------------------------------------------------
def print_leaderboard(df: pd.DataFrame, window_d: int) -> None:
    if df.empty:
        LOG.info("No results for %dd window", window_d)
        return
    top10 = df.sort_values("ret", ascending=False).head(10)
    print(f"\n=== Top performers — {window_d}-day window ===")
    print(
        top10[["model", "coin", "ret"]]
        .to_string(index=False, formatters={"ret": "{:+.1%}".format})
    )

# -------------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train missing models and back-test them across multiple windows."
    )
    parser.add_argument("--windows", nargs="*", type=int, default=DEFAULT_WINDOWS,
                        help="Look-back windows in days (e.g. 2 15 30)")
    parser.add_argument("--no-sizes", action="store_true", help="Ignore size_configs")
    parser.add_argument("--no-train", action="store_true", help="Skip auto-training")
    args = parser.parse_args()

    windows = sorted(set(args.windows))
    LOG.info("Requested windows: %s", windows)

    # 1. Load coins
    try:
        tokens_cfg = load_coins_yaml(COINS_YAML)
        LOG.info("Loaded %d coins from config", len(tokens_cfg))
    except Exception as e:
        LOG.error("Failed to load coins YAML: %s", e)
        sys.exit(1)

    # 2. Size overrides
    if not args.no_sizes:
        LOG.info("Applying size overrides…")
        for tk in list(tokens_cfg):
            sym = tokens_cfg[tk]["symbol"]
            tokens_cfg[tk] = merge_size_cfg(sym, tokens_cfg[tk])

    # 3. Backfill data
    try:
        ensure_backfill(tokens_cfg, max(windows))
    except Exception as e:
        LOG.error("Data backfill error: %s", e)
        sys.exit(1)

    # 4. Train champions
    LOG.info("Ensuring fresh models…")
    try:
        ensure_fresh_models(tokens_cfg, no_train=args.no_train)
    except Exception as e:
        LOG.error("Model training fatal: %s", e)
        sys.exit(1)

    # 5. Discover models
    all_paths = sorted(MODELS_DIR.glob("*.pkl"))
    models: List[tuple[str, int, Path]] = []
    for p in all_paths:
        if p.name.endswith("_sc.pkl"):
            continue
        token = _deduce_token(p.stem.split("_", 1)[0])
        if token not in tokens_cfg:
            LOG.debug("Skipping model %s: no config", p.name)
            continue
        models.append((token, parse_model_bar(p), p))
    LOG.info("Discovered %d models", len(models))

    # 6. Backtests
    all_rows: List[dict] = []
    for wd in windows:
        LOG.info("Starting backtests for %dd window…", wd)
        rows = run_backtests(models, tokens_cfg, wd)
        if not rows:
            LOG.info("No backtest results for %dd window", wd)
        else:
            df = pd.DataFrame(rows)
            print_leaderboard(df, wd)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = DATA_DIR / f"backtest_detail_{wd}d_{ts}.csv"
            df.to_csv(path, index=False)
            LOG.info("Detailed results for %dd window → %s", wd, path.name)
        all_rows.extend(rows)

    if not all_rows:
        LOG.error("No backtest results at all – exiting.")
        sys.exit(1)

    # 7. Cross-window summary
    summary_df = (
        pd.DataFrame(all_rows)
        .groupby(["model", "coin"], as_index=False)
        .agg(ret_mean=("ret", "mean"), dd_worst=("ret", "min"))
        .sort_values("ret_mean", ascending=False)
    )
    print("\n=== Cross-window summary ===")
    print(
        summary_df.head(20).to_string(index=False, formatters={"ret_mean": "{:+.1%}".format})
    )
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"backtest_summary_{ts}.csv"
    summary_df.to_csv(path, index=False)
    LOG.info("Summary CSV → %s", path.name)

    # 8. Show-me-the-money
    try:
        df_all = pd.DataFrame(all_rows)
        pivot = df_all.pivot_table(index=["coin", "model"], columns="window_d", values="ret")
        def pick_pnl(row):
            return row[15] if 15 in row.index and not np.isnan(row[15]) else row.max(skipna=True)
        pnl = pivot.apply(pick_pnl, axis=1).reset_index().rename(columns={0: "pnl_15d"})
        stats = df_all.groupby(["coin", "model"], as_index=False).agg(sharpe=("ret", "mean"), drawdown=("ret", "min"))
        best = pnl.merge(stats, on=["coin", "model"]).sort_values("pnl_15d", ascending=False).groupby("coin", as_index=False).first()
        print("\n=== SHOW ME THE MONEY ===")
        print(best.to_string(index=False, formatters={
            "pnl_15d": "{:+.1%}".format, "drawdown": "{:+.1%}".format, "sharpe": "{:.2f}".format
        }))
        log = json.loads(COIN_LOG.read_text()) if COIN_LOG.exists() else {}
        for _, r in best.iterrows():
            log[r["coin"]] = {
                "model":      r["model"],
                "pnl_15d":    float(r["pnl_15d"]),
                "sharpe":     float(r["sharpe"]),
                "drawdown":   float(r["drawdown"]),
                "ts":         ts,
            }
        COIN_LOG.write_text(json.dumps(log, indent=2))
        LOG.info("Coin performance log updated → %s", COIN_LOG.name)
    except Exception as e:
        LOG.error("Show-me-the-money fail: %s", e)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        LOG.error("Fatal: %s", exc)
        sys.exit(1)
