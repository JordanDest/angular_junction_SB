"""
 pipelines/trading.py
 ────────────────────
 Paper‑trading driver for one or many coins defined in config/coins.yaml.

 • Reads YAML → `coins` list
 • Loads per‑symbol *sizing* overrides in size_configs/<symbol>_best.json (if present)
 • Launches a KTrader instance per symbol
 • Pulls candles from DataManager.stream_candles()
 • Handles the “no model yet” case by waiting/re‑checking every 60 s
 """

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml
from core.ktrader import KTrader
from utils.data_manager import DataManager
from utils.model_factory import get_champion
from utils.utils import configure_logging, heartbeat

_LOG = logging.getLogger("Trading")


 # ──────────────────────────────────────────────────────────────────────────
 # Config helpers
 # ──────────────────────────────────────────────────────────────────────────
def load_coins(yaml_path: Path = Path("config/coins.yaml")) -> List[dict]:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["coins"]


def merge_size_cfg(symbol: str, coin_cfg: dict) -> dict:
    """Merge sizing overrides from size_configs/<symbol>_best.json, if it exists."""
    size_path = Path(f"size_configs/{symbol}_best.json")
    if size_path.exists():
        try:
            overrides = json.loads(size_path.read_text())
            if not isinstance(overrides, dict):
                raise TypeError("size config must be a JSON object")
            merged = {**coin_cfg, **overrides}
            _LOG.info("[%s] Loaded size overrides from %s", symbol, size_path)
            return merged
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("[%s] Failed to load %s – using defaults (%s)", symbol, size_path, exc)
    else:
        _LOG.warning("[%s] No size config found – using default sizing", symbol)
    return coin_cfg


# ──────────────────────────────────────────────────────────────────────────
# Candle stream shim
# ──────────────────────────────────────────────────────────────────────────
async def candle_stream(dm: DataManager, coin_cfg: dict):
    """Async generator of closed candles as pd.Series."""
    symbol = coin_cfg["symbol"]
    interval = coin_cfg.get("interval_minutes", 1)
    async for bar in dm.stream_candles(symbol, interval=interval):  # type: ignore[attr-defined]
        yield bar

# ──────────────────────────────────────────────────────────────────────────
# Trader task
# ──────────────────────────────────────────────────────────────────────────
async def trade_symbol(coin_cfg: dict, dm: DataManager, global_equity: List[float]):
    symbol = coin_cfg["symbol"]
    interval = coin_cfg.get("interval_minutes", 1)
     # Wait for a champion model
    model_path: Path | None = None
    while model_path is None:
        try:
            model_path = get_champion(symbol)  # expected to return Path | None
        except FileNotFoundError:
            _LOG.warning("[%s] No model yet; re‑checking in 60 s", symbol)
            await asyncio.sleep(60)
     # Apply per‑symbol sizing overrides BEFORE instantiating trader
    coin_cfg = merge_size_cfg(symbol, coin_cfg)
    trader = KTrader(coin_cfg, model_path=model_path, global_equity=global_equity)
    last_hb = time.time()
    async for candle in candle_stream(dm, coin_cfg):
        try:
            trader.step(candle)
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("[%s] step error: %s", symbol, exc)
         # heartbeat every 5 minutes
        if time.time() - last_hb >= 300:
            heartbeat("paper", symbol, trader.capital, trader.pos_units)
            last_hb = time.time()

# ──────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────
async def main():
    coins = load_coins()
    dm = DataManager(coins)  # starts no threads yet (unless you choose start_streaming)
    dm.backfill_if_missing()  # blocking but only first run is heavy
    dm.start_streaming(refresh_days=30)  # background threads
    equity = [sum(c.get("starting_capital", 1_000) for c in coins)]
    tasks = [
        asyncio.create_task(trade_symbol(cfg, dm, equity), name=cfg["symbol"])
        for cfg in coins
    ]
    await asyncio.gather(*tasks)


 # ------------------------------------------------------------------ CLI --#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper trading pipeline")
    configure_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _LOG.info("Interrupted by user")
        sys.exit(0)
