# file_store.py  – 100 % self‑contained
from __future__ import annotations
from pathlib import Path
import os, tempfile, threading
import pandas as pd
from datetime import datetime

###############################################################################
# Naming helpers
###############################################################################

def make_filename(symbol: str,
                  *,
                  days_back: int,
                  interval_minutes: int,
                  date: datetime | None = None) -> str:
    """
    Canonical file‑name builder – **the only** place that knows the format.

        <symbol>_<D>d_<M>m_<YYYY‑MM‑DD>.csv
    """
    date = date or datetime.utcnow()
    return f"{symbol}_{days_back}d_{interval_minutes}m_{date:%Y-%m-%d}.csv"

def make_path(data_dir: Path,
              symbol: str,
              *,
              days_back: int,
              interval_minutes: int,
              date: datetime | None = None) -> Path:
    return data_dir / make_filename(symbol,
                                    days_back=days_back,
                                    interval_minutes=interval_minutes,
                                    date=date)

###############################################################################
# Atomic writer
###############################################################################

_locks: dict[str, threading.Lock] = {}   # 1 lock per symbol

def atomic_write(df: pd.DataFrame,
                 out_path: Path) -> None:
    """
    Atomically replace *out_path* with *df* (creates parent dir if missing).
    Thread‑safe per‑symbol.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    symbol = out_path.stem.split('_', 1)[0]
    lock   = _locks.setdefault(symbol, threading.Lock())

    with lock:
        # 1. write to a tmp file in the same directory (same FS boundary)
        tmp = tempfile.NamedTemporaryFile("w",
                                          dir=out_path.parent,
                                          suffix=".csv",
                                          delete=False)
        try:
            df.to_csv(tmp, index=False)
            tmp.flush(); os.fsync(tmp.fileno())
        finally:
            tmp.close()

        # 2. atomic replace → no partial files, reader always sees full file
        Path(tmp.name).replace(out_path)
