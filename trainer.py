#!/usr/bin/env python3
"""
trainer.py – Full‐grid LSTM trainer with multi‐bar backfill + 15-day PnL pruning.

Place this file inside your `utils/` directory alongside:
  - data_manager.py
  - model_factory.py
  - backfill.py

Run it directly (e.g. `python trainer.py`) without needing `utils` to be a package.

For each coin in config/coins.yaml and each bar‐size in [1, 5, 10, 15, 30, 60, 120, 180, 360]:
  1. Ensure a fresh OHLC CSV exists via DataManager (Kraken /Trades backfill).
  2. Train every LSTM variant (horizon, seq_len, hidden, dropout, threshold) on that CSV.
  3. Run a 15-day backtest. Discard any model whose pnl_15d ≤ 0.
  4. Keep only the top K variants by pnl_15d; delete the rest.
  5. Record every kept variant’s hyperparameters + metrics in a master CSV.

Usage:
    cd path/to/StockFactory2/utils
    python trainer.py \
      [--coins-yaml <path>] \
      [--data-dir <path>] \
      [--days-back <N>] \
      [--preset <micro|scalp|swing>] \
      [--keep-top-k <K>] \
      [--verbose] \
      [--fast-train]

Example:
    cd c:/Users/jordd/OneDrive/Desktop/Code/StockBot/StockFactory2/utils
    python trainer.py \
      --coins-yaml ../config/coins.yaml \
      --data-dir ../data \
      --days-back 55 \
      --preset scalp \
      --keep-top-k 5 \
      --verbose
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import yaml
import pandas as pd
#from textTest import send_text
# ---------------------------------------------------------------
# Ensure that "utils/" (this folder) is on sys.path so we can import
# data_manager.py and model_factory.py directly without needing __init__.py.
FILE = Path(__file__).resolve()
UTILS_DIR = FILE.parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
# ---------------------------------------------------------------

from utils.data_manager import DataManager                                 # :contentReference[oaicite:0]{index=0}
from utils.model_factory import (                                          # :contentReference[oaicite:1]{index=1}
    build_hyper_grid,
    train_variant,
    quick_backtest_15d,
    serialize_model_scaler,
)

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = UTILS_DIR.parent
LOG_PATH = PROJECT_ROOT / "trainer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
log = logging.getLogger("trainer")


def load_coins(cfg_path: Path) -> List[Dict[str, Any]]:
    """
    Parse config/coins.yaml into a list of dicts.
    Expects a top-level "coins:" list where each entry has at least "symbol".
    Returns a list like: [ { "symbol": "XXBTZUSD" }, { "symbol": "XETHZUSD" }, … ]
    """
    if not cfg_path.exists():
        raise FileNotFoundError(f"Coins configuration missing: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not raw or "coins" not in raw:
        raise ValueError("'coins.yaml' must contain a top-level 'coins:' list")
    coins = raw["coins"]
    if not isinstance(coins, list) or len(coins) == 0:
        raise ValueError("'coins.yaml' lists zero coins – nothing to do")
    for entry in coins:
        if not isinstance(entry, dict) or "symbol" not in entry:
            raise ValueError("Every coin entry needs a 'symbol' key")
        entry["symbol"] = entry["symbol"].upper().strip()
    return coins

# ────────────────────────────────────────────────────────────────────
# Champion promotion helper
# ────────────────────────────────────────────────────────────────────
def promote_coin_champion(
    symbol: str,
    master_log: List[Dict[str, Any]],
    models_dir: Path,
    metric: str = "pnl_15d",
) -> None:
    """
    From *master_log* pick the row with the highest *metric* for *symbol*,
    rename its files to   <symbol>_champion.*,
    delete every other model belonging to that symbol.
    """
    cand = [r for r in master_log if r["symbol"] == symbol]
    if not cand:
        log.warning("%s – nothing to promote", symbol); return

    best = max(cand, key=lambda r: r.get(metric, -9e9))
    champ_src = Path(best["model_path"])
    champ_base = models_dir / f"{symbol.lower()}_champion"

    # rename (atomic on same FS)
    champ_src.replace(champ_base.with_suffix(".pkl"))
    for ext in ("_scaler.pkl", ".json"):
        src = champ_src.with_name(champ_src.stem + ext)
        if src.exists():
            src.replace(champ_base.with_suffix(ext))

    # purge every other file for this coin
    for row in cand:
        if row is best:
            continue
        stem = Path(row["model_path"]).stem
        for ext in (".pkl", "_scaler.pkl", ".json"):
            junk = models_dir / f"{stem}{ext}"
            junk.unlink(missing_ok=True)

    log.info("%s – champion promoted → %s", symbol, champ_base.with_suffix('.pkl').name)
    
def train_and_prune(
    symbol: str,
    csv_path: Path,
    bar: int,
    preset: str,
    keep_top_k: int,
    min_pnl: float,
    output_dir: Path,
    fast_train: bool,
    master_log: List[Dict[str, Any]],
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    1) Load CSV → DataFrame (time-indexed).
    2) Build full hyper-grid for this (bar, preset).
    3) Train each variant, run 15-day backtest.
    4) Discard if pnl_15d ≤ min_pnl.
    5) Serialize model + scaler + JSON for each kept variant.
    6) Keep only top_k variants by pnl_15d; delete the rest from disk.
    7) Append each kept record to master_log.

    Returns a list of (model_path, record_dict) for the kept variants.
    """
    log.info(f"    ▶ Reading CSV for {symbol} @ {bar}m: {csv_path.name}")
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")

    # The CSV was already gap-filled & sanitized by DataManager._perform_backfill.
    # We assume no NaNs remain. If you want, do a sanity check:
    if df.isnull().any().any():
        log.warning(f"      ↳ CSV {csv_path.name} contains NaNs. Dropping rows with NaN.")
        df = df.dropna()

    # 1) Build hyperparameter grid for this bar + preset
    log.info(f"    ▶ Building hyperparameter grid: bar={bar}m, preset={preset}")
    hyper_grid = build_hyper_grid(preset=preset, bar=bar)

    kept_variants: List[Tuple[Path, Dict[str, Any]]] = []

    for hp in hyper_grid:
        horizon = hp["horizon"]
        seq_len = hp["seq_len"]
        hidden = hp["hidden"]
        dropout = hp["dropout"]
        thr = hp["label_threshold"]

        log.debug(f"      ▶ HP: horizon={horizon}, seq_len={seq_len}, hidden={hidden}, "
                  f"dropout={dropout}, threshold={thr}")
        try:
            model, scaler, metrics = train_variant(
                df=df,
                hp=hp,
                fast_train=fast_train,
            )
        except Exception as e:
            log.warning(f"        ↳ {symbol}@{bar}m HP={hp} failed during training: {e}")
            continue

        f1_test = metrics.get("f1_test", 0.0)
        pr_auc_test = metrics.get("pr_auc_test", 0.0)

        # 2) 15-day backtest on unseen tail
        try:
            pnl_15d = quick_backtest_15d(
                model=model,
                scaler=scaler,
                df=df,
                horizon=horizon,
            )
        except Exception as e:
            log.warning(f"        ↳ {symbol}@{bar}m HP={hp} failed 15d backtest: {e}")
            continue

        # 3) Gate: discard if pnl_15d ≤ min_pnl
        if pnl_15d <= min_pnl:
            log.info(f"        ↳ Rejected: pnl_15d={pnl_15d:.2f} ≤ {min_pnl:.2f}")
            continue

        # 4) Serialize model, scaler, and JSON record
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname_base = (
            f"{symbol}_b{bar}_h{horizon}_sl{seq_len}_hid{hidden}"
            f"_dp{dropout}_thr{thr:.3f}_{timestamp}"
        )
        model_path = output_dir / f"{fname_base}.pkl"
        scaler_path = output_dir / f"{fname_base}_scaler.pkl"
        json_path = output_dir / f"{fname_base}.json"

        try:
            serialize_model_scaler(model, scaler, model_path, scaler_path)
        except Exception as e:
            log.warning(f"        ↳ Failed to serialize model/scaler: {e}")
            continue

        record: Dict[str, Any] = {
            "symbol": symbol,
            "bar": bar,
            "horizon": horizon,
            "seq_len": seq_len,
            "hidden": hidden,
            "dropout": dropout,
            "label_threshold": thr,
            "f1_test": f1_test,
            "pr_auc_test": pr_auc_test,
            "pnl_15d": pnl_15d,
            "ts": datetime.utcnow().isoformat() + "Z",
            "model_path": str(model_path.resolve()),
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            yaml.safe_dump(record, jf)

        log.info(
            f"        ↳ Kept: pnl_15d={pnl_15d:.2f}, f1={f1_test:.4f}, pr_auc={pr_auc_test:.4f} → {model_path.name}"
        )
        kept_variants.append((model_path, record))
        LOG_FILE = PROJECT_ROOT / "model_master_log.csv"
        pd.DataFrame([record]).to_csv(
            LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False
        )

    # 5) Keep only top_k variants by pnl_15d
    kept_variants.sort(key=lambda x: x[1]["pnl_15d"], reverse=True)
    to_keep = kept_variants[:keep_top_k]
    to_delete = kept_variants[keep_top_k:]

    for mdl_path, rec in to_delete:
        frag = mdl_path.stem  # base filename without .pkl
        scaler_file = mdl_path.with_name(f"{frag}_scaler.pkl")
        json_file = mdl_path.with_suffix(".json")
        for f in (mdl_path, scaler_file, json_file):
            if f.exists():
                try:
                    f.unlink()
                except Exception as e:
                    log.warning(f"          ↳ Failed to delete {f.name}: {e}")

    # 6) Append each kept record to master_log
    for _, rec in to_keep:
        master_log.append(rec)

    return to_keep


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full‐grid LSTM trainer with multi‐bar backfill + 15-day PnL pruning"
    )
    ap.add_argument(
        "--coins-yaml",
        type=Path,
        default=PROJECT_ROOT / "config" / "coins.yaml",
        help="Path to coins.yaml (default: %(default)s)",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory where OHLC CSVs live / will be written (default: %(default)s)",
    )
    ap.add_argument(
        "--days-back",
        type=int,
        default=55,
        help="How many days to backfill if CSV is missing (default: %(default)d)",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="scalp",
        choices=["micro", "scalp", "swing"],
        help="Which hyper‐param 'preset' to use (default: %(default)s)",
    )
    ap.add_argument(
        "--keep-top-k",
        type=int,
        default=5,
        help="How many top variants by pnl_15d to keep per (symbol, bar) (default: %(default)d)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    ap.add_argument(
        "--fast-train",
        action="store_true",
        help="Set FAST_TRAIN=1 to do a single-variant smoke test",
    )

    args = ap.parse_args()

    if args.fast_train:
        os.environ["FAST_TRAIN"] = "1"

    # Adjust log level if verbose
    if args.verbose:
        log.setLevel(logging.DEBUG)

    # 1) Load coin list
    try:
        coins = load_coins(args.coins_yaml)
    except Exception as e:
        log.error("Failed to load coins YAML: %s", e)
        sys.exit(1)
    log.info("Loaded %d coins from %s", len(coins), args.coins_yaml)

    # 2) Instantiate DataManager
    dm = DataManager(
        coin_settings=coins,
        data_directory=args.data_dir,
        default_days_back=args.days_back,
        verbose=args.verbose,
        interactive=False,
    )
    log.info("Initialized DataManager with data_dir=%s", args.data_dir)
    # try:
    #     send_text(body = "Training sequence initialized", subject = "StockBot: ")
    # finally:
    #     pass
    # 3) Define the bar‐sizes to sweep (minutes)
    BAR_SIZES = [1, 5, 10, 15, 30, 60, 120, 180, 360]

    # 4) Prepare output directory for models
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    master_log: List[Dict[str, Any]] = []
    overall_failures: Dict[str, str] = {}

    # 5) Loop over coins and bars
    for coin_cfg in coins:
        symbol = coin_cfg["symbol"]
        log.info(f"▶ Starting sweeps for {symbol} …")
        # try:
        #     send_text(body = f"Starting sweeps for {symbol} ", subject = "StockBot: ")
        # finally:
        #     pass
        for bar in BAR_SIZES:
            log.info(f"  → Preparing backfill for {symbol} @ {bar}m …")
            # Construct a temporary coin_cfg to force backfill at this bar-size
            temp_cfg = {
                "symbol": symbol,
                "interval_minutes": bar,
                "days_back": args.days_back,
            }

            # Perform backfill (Kraken /Trades → OHLC → Gap-Fill → Validate → CSV)
            try:
                dm._perform_backfill(temp_cfg)
            except Exception as e:
                msg = f"{symbol}@{bar}m: backfill failed: {e}"
                log.error(msg)
                overall_failures[f"{symbol}@{bar}m"] = str(e)
                continue

            # Get the CSV path that was just written
            try:
                date_str = datetime.utcnow().strftime("%m_%d_%Y")
                csv_filename = f"{symbol}_{temp_cfg['days_back']}d_{bar}m_{date_str}.csv"
                csv_path = dm.get_csv(symbol, interval_minutes=bar, days_back=args.days_back)
            except Exception as e:
                msg = f"{symbol}@{bar}m: failed to derive CSV path: {e}"
                log.error(msg)
                overall_failures[f"{symbol}@{bar}m"] = str(e)
                continue

            if not csv_path.exists():
                msg = f"{symbol}@{bar}m: CSV not found at {csv_path}"
                log.error(msg)
                overall_failures[f"{symbol}@{bar}m"] = msg
                continue

            # 6) Train + prune variants on that CSV
            log.info(f"  → Training grid for {symbol} @ {bar}m …")
            from model_factory import train_grid

            # build a full config dict including this bar size
            bar_cfg = {**coin_cfg, "interval_minutes": bar}
            try:
               kept_paths = train_grid(
                   bar_cfg,
                   csv_path,
                   bars=[bar],
                   preset=args.preset,
                   depth="partial" if os.getenv("FAST_TRAIN") == "1" else "full",
               )
            except Exception as e:
                msg = f"{symbol}@{bar}m: train_grid failed: {e}"
                log.error(msg)
                overall_failures[f"{symbol}@{bar}m"] = str(e)
                continue

            for p in kept_paths:
                meta = yaml.safe_load((p.with_suffix(".json")).read_text())
                master_log.append(meta)
                log.info("    → Kept model: %s (bar=%dm)", p.name, meta["bar"])

    # 7) Shutdown DataManager
    dm.stop()

    # 8) Write master log to CSV
    try:
        df_master = pd.DataFrame(master_log)
        df_master.to_csv(PROJECT_ROOT / "model_master_log.csv", index=False)
        log.info("Wrote model_master_log.csv with %d entries.", len(master_log))
    except Exception as e:
        log.error("Failed to write master log: %s", e)

    # 9) Report failures (if any)
    if overall_failures:
        log.error(
            "%d/%d (symbol,bar) combinations failed – see trainer.log for details.",
            len(overall_failures),
            len(coins) * len(BAR_SIZES),
        )
        for combo, reason in overall_failures.items():
            log.error("%s: %s", combo, reason)
        sys.exit(1)

    print("\n✅ All training sweeps finished successfully.")
    print(f"Models are under: {MODELS_DIR.resolve()}")
    print(f"Master log: {PROJECT_ROOT / 'model_master_log.csv'}")


if __name__ == "__main__":
    main()
