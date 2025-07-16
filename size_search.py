from itertools import product
from pathlib import Path
from core.ktrader import KTrader
from utils.data_manager import DataManager

SIZE_GRID = {   # shrink / expand as you like
    "kelly_fraction":   [0.25, 0.5, 0.75, 1.0],
    "risk_frac_base":   [0.001, 0.0025, 0.005],
    "stop_loss_atr_mult": [1.5, 2.0, 2.5],
    "trail_stop_pct":   [0.005, 0.01],
}

def search_sizes(symbol: str, model_path: Path, coin_cfg: dict, candles):
    keys, values = zip(*SIZE_GRID.items())
    best = None
    for combo in product(*values):
        cfg = {**coin_cfg, **dict(zip(keys, combo))}
        trader = KTrader(cfg, model_path, global_equity=[100.0])
        for row in candles.itertuples(index=False):
            trader.step(row._asdict())
        sharpe = trader.sharpe()
        dd     = trader.max_drawdown()
        if best is None or (sharpe, -dd) > (best[0], -best[1]):
            best = (sharpe, dd, cfg)
    return best[-1]   # cfg
