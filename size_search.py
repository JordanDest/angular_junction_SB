"""
Trading sizing-grid back-test runner (importable API).

Exports:
    search_sizes(symbol, champion, cfg, df_recent, max_combos=1000) -> best_override dict

Internals:
    _search_coin(), _run_combo(), SIZE_GRID, etc.
"""

from __future__ import annotations
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import random
import numpy as np
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice, product
from tqdm.auto import tqdm

# -------------------------------------------------------------------------------------
# Static paths (for writing out JSON and loading coin configs)
# -------------------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]  # StockFactory2/
CONFIG_YAML = ROOT / "config" / "coins.yaml"
SIZE_DIR    = ROOT / "size_configs"

SIZE_DIR.mkdir(exist_ok=True)

_LOG = logging.getLogger("size_grid")

# -------------------------------------------------------------------------------------
# Helper – load coins.yaml
# -------------------------------------------------------------------------------------
def _load_coins(path: Path = CONFIG_YAML) -> Dict[str, dict]:
    data = yaml.safe_load(path.read_text())
    coins = data["coins"] if isinstance(data, dict) else data
    return {c["symbol"]: c for c in coins}

# -------------------------------------------------------------------------------------
# Default sizing-grid (tweak as needed)
# -------------------------------------------------------------------------------------
SIZE_GRID = {
    "risk_frac_base":     [0.001, 0.0025, 0.005],
    "kelly_fraction":     [0.25, 0.5, 0.75, 1.0],
    "stop_loss_atr_mult": [1.5, 2.0, 2.5],
    "trail_stop_pct":     [0.005, 0.01, 0.015],
    "take_profit_pct":    [0.015, 0.025, 0.035],
    "break_even_pct":     [0.003, 0.005],
}

# -------------------------------------------------------------------------------------
# Helper – discover all models of a symbol (unused in search_sizes(), but kept for
# backward compatibility if someone imports it)
# -------------------------------------------------------------------------------------
def _discover_models(symbol: str) -> List[Path]:
    MODELS_DIR = ROOT / "models"
    sym_low = symbol.lower()
    return [
        p for p in MODELS_DIR.glob("*.pkl")
        if sym_low in p.stem.lower() and not p.name.endswith("_sc.pkl")
    ]

# -------------------------------------------------------------------------------------
# Compute performance metrics for one equity curve
# -------------------------------------------------------------------------------------
def _metrics(equity: np.ndarray) -> Tuple[float, float, float]:
    if equity.size < 2:
        return 0.0, 0.0, 0.0
    total_ret = equity[-1] / equity[0] - 1
    rets      = np.diff(equity) / equity[:-1]
    dd        = (equity / np.maximum.accumulate(equity) - 1).min()
    sharpe    = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(1440)
    return float(total_ret), float(dd), float(sharpe)

# -------------------------------------------------------------------------------------
# Multiprocess work: run one param‐combo on one model + list-of‐records
# -------------------------------------------------------------------------------------
def _run_combo(
    records: List[dict],
    symbol: str,
    model_path: str,
    starting_cap: float,
    overrides: dict
) -> Tuple[dict, Tuple[float, float, float]]:
    # Import locally to avoid pulling in torch/TensorFlow prematurely
    from core.ktrader import KTrader

    trader = KTrader(
        { "symbol": symbol, **overrides },
        Path(model_path),
        [starting_cap],
        starting_capital=starting_cap
    )
    equity = [starting_cap]
    for bar in records:
        trader.step(bar)
        equity.append(trader.capital)
    return overrides, _metrics(np.array(equity, dtype=float))

# -------------------------------------------------------------------------------------
# Run a full grid search for one coin + one model + a given records list
# -------------------------------------------------------------------------------------
def _search_coin(
    symbol: str,
    model_path: Path,
    records: List[dict],
    max_combos: int = 1000
) -> Tuple[dict, pd.DataFrame]:
    """
    Returns (best_override_dict, full_results_dataframe).  Columns of df:
    [each key in SIZE_GRID] + return, drawdown, sharpe, utility.
    """
    # 1) Build all possible combos, or truncate if max_combos < total
    keys, choices = zip(*SIZE_GRID.items())
    combo_iter = (dict(zip(keys, vals)) for vals in product(*choices))

    if max_combos and max_combos < math.prod(len(c) for c in choices):
        combo_iter = islice(combo_iter, max_combos)

    starting_cap = 100.0
    rows = []
    with ProcessPoolExecutor(max_workers=min(8, max_combos)) as pool:
        futures = {
            pool.submit(_run_combo, records, symbol, str(model_path), starting_cap, combo): combo
            for combo in combo_iter
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{symbol} combos"):
            combo = futures[fut]
            try:
                _, (ret, dd, shr) = fut.result()
                rows.append({ **combo, "return": ret, "drawdown": dd, "sharpe": shr })
            except Exception as exc:
                _LOG.exception("[%s] combo failed: %s", symbol, exc)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No successful runs for {symbol}")

    # 2) Rank by (sharpe, -drawdown, return) lex order
    df["utility"] = df.apply(lambda r: (r.sharpe, -r.drawdown, r["return"]), axis=1)
    df = df.sort_values("utility", ascending=False, key=lambda s: s.apply(tuple))

    # 3) Pick best override dictionary
    best = df.iloc[0][list(SIZE_GRID.keys())].to_dict()
    best["return"] = df.iloc[0]["return"]
    return best, df

# -------------------------------------------------------------------------------------
# PUBLIC API: use this from orchestrator or main.py
# -------------------------------------------------------------------------------------
def search_sizes(
    symbol: str,
    champion: Any,
    cfg: dict,
    df_recent: pd.DataFrame,
    max_combos: int = 1000
) -> dict:
    """
    Run sizing‐grid backtest on exactly one champion model + df_recent:

      • `symbol`:     e.g. "BTC"
      • `champion`:   object with `.path` attr or a string path
      • `cfg`:        coin config from coins.yaml (unused here unless you want to sniff hyperparams)
      • `df_recent`:  pd.DataFrame of minute‐bars (already filtered to last N days)
      • `max_combos`: cap on grid permutations (default 1000, pass 0 to run full grid)

    Returns a `best_override` dict that always has at least:
      {
        "risk_frac_base": …,
        "kelly_fraction": …,
        "stop_loss_atr_mult": …,
        "trail_stop_pct": …,
        "take_profit_pct": …,
        "break_even_pct": …,
        "return": …              # total PnL over the backtest period
      }

    Also writes `size_configs/<symbol>_best.json` with the same dict.
    """
    # Coerce champion → filesystem path
    if hasattr(champion, "path"):
        model_path = Path(champion.path)
    else:
        model_path = Path(str(champion))

    if not model_path.exists():
        raise FileNotFoundError(f"Champion model file not found for {symbol}: {model_path}")

    # Convert df_recent → list-of‐records
    df_copy = df_recent.copy()
    if "time" in df_copy.columns:
        df_copy["time"] = pd.to_datetime(df_copy["time"], utc=True)
    records = df_copy.sort_values("time").to_dict("records")

    # Run the private combo search
    best_override, _ = _search_coin(symbol, model_path, records, max_combos=max_combos)

    # Persist best_override JSON
    SIZE_DIR.mkdir(exist_ok=True)
    out_path = SIZE_DIR / f"{symbol}_best.json"
    out_path.write_text(json.dumps(best_override, indent=2))

    return best_override

# -------------------------------------------------------------------------------------
# (OPTIONAL) Keep a CLI entrypoint if you ever need “full universe” tests
# -------------------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sizing grid-search back-test (CLI)")
    parser.add_argument("--days", type=int, default=30, help="Look-back window in days")
    parser.add_argument("--max-combos", type=int, default=1000, help="Limit grid permutations (0 = full)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # If someone runs `python -m pipelines.size_search --days 30 --max-combos 1000`,
    # we can fallback to the old full-universe mode:
    coins_cfg = _load_coins()  # expects coins.yaml
    summary_rows = []
    for symbol, cfg in coins_cfg.items():
        # discover random model (like before)
        models = _discover_models(symbol)
        if not models:
            _LOG.warning("No models for %s – skipping", symbol)
            continue
        model_path = random.choice(models)
        _LOG.info("[%s] Using model %s", symbol, model_path.name)

        # load CSV, filter last args.days, call _search_coin(...)
        # … [ replicate old logic, or remove if CLI not needed ] …

    # (old summary→CSV logic here, if desired)

if __name__ == "__main__":
    main()
