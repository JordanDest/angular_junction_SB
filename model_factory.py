from __future__ import annotations
"""ModelFactory – unified, pickle-safe model-training pipeline (refactored).

Key fixes / enhancements (2025-06-05)
-------------------------------------
* **Dynamic ``FAST_TRAIN`` & ``DEVICE``** – read at *runtime* from env-vars.
* **Variable ``seq_len``** – honour the value coming from the hyper-grid
  instead of the former global constant; removes a silent tuning bug.
* **Less-verbose auto-threshold** – the info log appears **once per symbol**
  (subsequent variants use cached value and emit only a debug line).
* **Temp-dir cleanup** – context-managed checkpoint directory.
* Minor readability tweaks & dead-code pruning (removed old shim body).
"""

###############################################################################
# Imports & paths
###############################################################################

import json
import logging
import os
import random
import re
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Literal

import joblib
import numpy as np
import pandas as pd
import torch
from fastai.callback.all import (
    CSVLogger,
    EarlyStoppingCallback,
    SaveModelCallback,
)
from fastai.callback.core import Callback
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from fastai.learner import CancelFitException, Learner
from fastai.metrics import (
    AccumMetric,
    F1Score,
    Precision,
    Recall,
    RocAucBinary,
    accuracy,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset, WeightedRandomSampler

# Project-level helpers (provide your own stubs if they don't exist)
from utils.data_manager import _fill_missing_buckets, _sanitize_timestamps  # type: ignore
from utils.utils import (
    CAPITAL,
    get_logger,
    quick_score as _quick_score,
    stop_flag,
)  # type: ignore

# Silence sklearn's precision==0 warnings during threshold search
import warnings
import sklearn.exceptions as skex

warnings.filterwarnings("ignore", category=skex.UndefinedMetricWarning)

###############################################################################
# Globals – overridable via env vars
###############################################################################

DEFAULT_SEQ_LEN = 60
EPOCHS = int(os.getenv("EPOCHS", 10))
LR = float(os.getenv("LR", 1e-3))
RNG_SEED = 42

# Fast-train is controlled *at runtime* by the env var
FAST_TRAIN = bool(int(os.getenv("FAST_TRAIN", "0")))

KEEP_TOP = 4
KEEP_ALL_MODELS = True
MIN_F1 = 0.60
KEEP_BASE, KEEP_MAX, KEEP_DELTA = 4, 7, 0.02

# ----------------------------------------------------------------------------
# Device selection honouring the optional DEVICE env-var
# ----------------------------------------------------------------------------
_ENV_DEVICE = os.environ.get("DEVICE", "").strip().lower()
if _ENV_DEVICE:
    DEVICE = (
        "cuda"
        if _ENV_DEVICE.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# Paths & logging
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BASE_LOG_DIR = PROJECT_ROOT / "logs"
BASE_LOG_DIR.mkdir(exist_ok=True)

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
LOG_DIR = BASE_LOG_DIR / RUN_ID
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG = get_logger("ModelFactory")
_LOG.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8", mode="a")
fh.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
)
_LOG.addHandler(fh)

###############################################################################
# Hyper-parameter grid (ordered) – note the added *seq_len*
###############################################################################

# Static grid (base).  “seq_len” is now explicitly in HYPER_GRID instead of DEFAULT_SEQ_LEN.
HYPER_GRID: Dict[str, Sequence] = {
    "hidden": [32, 64, 128],
    "dropout": [0.1, 0.2, 0.4],
    "label_threshold": [0.008, 0.012, 0.015],
    "horizon": [20, 30],
    "seq_len": [30, 60, 120],
}

# Support for presets (dynamic extension of HYPER_GRID)
BARS = [1, 3, 5, 8, 15, 30, 60, 120, 240, 360]  # minutes
PRESETS: dict[str, dict[str, Sequence[int]]] = {
    "micro": {"horizon": [5, 10, 15], "seq_len": [10, 20, 30]},
    "scalp": {"horizon": [20, 30, 45], "seq_len": [30, 60, 120]},
    "swing": {"horizon": [60, 120, 240], "seq_len": [120, 240, 480, 720]},
}

# def _build_hyper_grid(cfg: dict, *, preset: str) -> dict[str, Sequence]:
#     if preset not in PRESETS:
#         _LOG.warning("Unknown preset %s – falling back to static grid", preset)
#         return HYPER_GRID.copy()

#     dyn = dict(HYPER_GRID)
#     dyn.update(PRESETS[preset])
#     dyn["seq_len"] = [s for s in dyn["seq_len"] if s <= 1000]
#     dyn["horizon"] = [h for h in dyn["horizon"] if h < min(dyn["seq_len"])]
#     if not dyn["horizon"] or not dyn["seq_len"]:
#         raise ValueError("Dynamic grid collapsed – adjust preset ranges")
#     return dyn
def _build_hyper_grid(
    cfg: dict,
    *,
    preset: str = "scalp",
    bar: int | None = None,
) -> dict[str, Sequence]:
    """
    Internal helper that converts *preset* → concrete hyper‑parameter lists.

    Parameters
    ----------
    cfg : dict
        Coin / training config; only `interval_minutes` is read.
    preset : {"micro","scalp","swing"}
        Named grid template.
    bar : int | None
        Bar length **in minutes**.  If ``None`` we fall back to
        ``cfg.get("interval_minutes", 1)``.

    Returns
    -------
    dict[str, Sequence]
        Keys: hidden, dropout, label_threshold, horizon, seq_len

    Notes
    -----
    • When `bar > 1`, minute‑based *horizon* & *seq_len* are divided by `bar`
      so they remain the same **look‑ahead / look‑back in time**.
    • Ensures horizon  <  seq_len; raises if the grid collapses.
    """
    if preset not in PRESETS:
        _LOG.warning("Unknown preset %s – falling back to static grid", preset)
        base = HYPER_GRID.copy()
    else:
        base = {**HYPER_GRID, **PRESETS[preset]}

    bar = bar or cfg.get("interval_minutes", 1)

    def _scale(values: Sequence[int]) -> list[int]:
        return [max(1, v / bar) for v in values] if bar > 1 else list(values)

    grid = base.copy()
    grid["seq_len"] = _scale(grid["seq_len"])
    max_seq = min(grid["seq_len"])
    grid["horizon"] = [h for h in _scale(grid["horizon"]) if h <= max_seq]

    if not grid["seq_len"] or not grid["horizon"]:
        raise ValueError(
            f"Grid collapsed for bar={bar}.  "
            "Adjust PRESETS or HYPER_GRID so horizon < seq_len."
        )
    return grid

def _select_top(
    res: list[tuple[float, Path]],
    base_keep: int = KEEP_BASE,
    max_keep: int = KEEP_MAX,
    delta: float = KEEP_DELTA,
) -> list[Path]:
    if not res:
        return []
    res.sort(key=lambda t: t[0], reverse=True)
    champ_score = res[0][0]
    winners = [p for f1, p in res if f1 >= champ_score - delta]
    # keep the top `base_keep` by raw position, plus any within delta (up to max_keep total)
    keep_list: list[tuple[float, Path]] = res[:base_keep]
    extras = [(champ_score - 0.001, p) for p in winners[base_keep : max_keep]]
    keep_list.extend(extras)
    return [p for _, p in keep_list]

###############################################################################
# Misc helpers
###############################################################################

def _extract_f1(path: Path) -> float:
    m = re.search(r"_f1_(\d+\.\d+)", path.stem)
    return float(m.group(1)) if m else -1.0

def _quick_score_loud(symbol: str, mpath: Path) -> dict:
    res = _quick_score(symbol, mpath)
    if res is None:
        raise ValueError("quick_score returned None – back-test failed")
    return res if isinstance(res, dict) else {"pnl": float(res)}

def _needs_rebalance(y: np.ndarray, *, min_pos: int = 30, max_ratio: int = 10) -> bool:
    c0, c1 = np.bincount(y, minlength=2)
    return c1 < min_pos or (c0 / max(c1, 1)) > max_ratio

###############################################################################
# Losses, datasets, nets
###############################################################################

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, targets):  # type: ignore[override]
        ce = cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class _SeqDS(Dataset):
    """Sliding-window dataset taking the sequence length as ctor arg."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X, self.y, self.seq_len = X, y, seq_len

    def __len__(self) -> int:
        return max(len(self.X) - self.seq_len + 1, 0)

    def __getitem__(self, idx: int):
        if idx < 0 or idx + self.seq_len - 1 >= len(self.y):
            raise IndexError("SeqDS index out of range")
        xx = torch.tensor(self.X[idx : idx + self.seq_len], dtype=torch.float32)
        yy = torch.tensor(self.y[idx + self.seq_len - 1], dtype=torch.long)
        return xx, yy

    def new_empty(self):
        return self.__class__(
            np.empty((0, *self.X.shape[1:]), dtype=self.X.dtype),
            np.empty(0, dtype=self.y.dtype),
            self.seq_len,
        )

class _LSTM(torch.nn.Module):
    def __init__(self, inp: int, hidden: int, dropout: float):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            inp, hidden, num_layers=2, batch_first=True, dropout=dropout
        )
        self.norm = torch.nn.LayerNorm(hidden)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x):  # type: ignore[override]
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(torch.relu(self.norm(last)))
###############################################################################
# Callbacks – pickle‑safe
###############################################################################

class SaveBestCPU(SaveModelCallback):
    """SaveModelCallback that always checkpoints to CPU and never reloads."""

    def _save(self, name):
        dev0 = next(self.learn.model.parameters()).device
        if dev0.type != "cpu":
            self.learn.model.cpu()
        try:
            super()._save(name)
        finally:
            if dev0.type != "cpu":
                self.learn.model.to(dev0)

    def after_fit(self, **_):  # fastai signature compliance
        return  # skip automatic reload


class StopFlagCallback(Callback):
    """Graceful early exit when stop_flag is raised (Ctrl‑C friendly)."""

    order = 0

    def after_batch(self, **_):
        if stop_flag.is_set():
            raise CancelFitException()

    def after_epoch(self, **_):
        if stop_flag.is_set():
            raise CancelFitException()

# ###############################################################################
# # Dataset & model definitions
# ###############################################################################

# class _SeqDS(Dataset):
#     """Sliding‑window dataset."""

#     def __init__(self, X: np.ndarray, y: np.ndarray):
#         self.X, self.y = X, y

#     def __len__(self):
#         return max(len(self.X) - SEQ_LEN + 1, 0)

#     def __getitem__(self, idx: int):
#         if idx < 0 or idx + SEQ_LEN - 1 >= len(self.y):
#             raise IndexError("SeqDS index out of range")
#         xx = torch.tensor(self.X[idx : idx + SEQ_LEN], dtype=torch.float32)
#         yy = torch.tensor(self.y[idx + SEQ_LEN - 1], dtype=torch.long)
#         return xx, yy

#     def new_empty(self):
#         return self.__class__(
#             np.empty((0, *self.X.shape[1:]), dtype=self.X.dtype),
#             np.empty(0, dtype=self.y.dtype),
#         )


# class _LSTM(torch.nn.Module):
#     def __init__(self, inp: int, hidden: int, dropout: float):
#         super().__init__()
#         self.rnn = torch.nn.LSTM(inp, hidden, num_layers=2, batch_first=True, dropout=dropout)
#         self.norm = torch.nn.LayerNorm(hidden)
#         self.head = torch.nn.Linear(hidden, 2)

#     def forward(self, x):  # type: ignore[override]
#         out, _ = self.rnn(x)
#         last = out[:, -1, :]
#         return self.head(torch.relu(self.norm(last)))
###############################################################################
# Auto-threshold with simple per-symbol cache to cut log spam
###############################################################################

_AUTO_THR_CACHE: dict[tuple[str, int], float] = {}

def _auto_threshold(
    df: pd.DataFrame,
    *,
    target_ratio: float = 0.15,
    horizon: int = 30,
    guess: float = 0.01,
    sym: str,
) -> float:
    key = (sym, horizon)
    if key in _AUTO_THR_CACHE:
        _LOG.debug("[%s] cached label_threshold %.5f", sym, _AUTO_THR_CACHE[key])
        return _AUTO_THR_CACHE[key]

    lo, hi = guess / 10, guess * 10
    thr = guess
    for _ in range(10):
        thr = (lo + hi) / 2
        pos = (
            (df["high"].rolling(horizon).max().shift(-horizon) - df["mid"]) >= thr
        ).astype(int)
        ratio = pos.mean()
        if abs(ratio - target_ratio) < 0.01:
            break
        if ratio > target_ratio:
            lo = thr
        else:
            hi = thr

    _AUTO_THR_CACHE[key] = thr
    _LOG.info("[%s] auto label_threshold -> %.5f", sym, thr)
    return thr

###############################################################################
# Data preparation
###############################################################################

def _prepare(df: pd.DataFrame, cfg: dict):
    """Return train/val/test splits + fitted scaler & feature list."""
    int_min = cfg.get("interval_minutes", 1)
    sym = cfg["symbol"]
    horizon = cfg["horizon"]

    df = df.copy()
    if "time" in df.columns:
        df["time"] = _sanitize_timestamps(df["time"])
        df, n_fill = _fill_missing_buckets(df, interval_seconds=int_min * 60)
        if n_fill:
            _LOG.info("[%s] filled %d missing candles", sym, n_fill)

    df["mid"] = (df["high"] + df["low"]) / 2
    feats = ["mid", "high", "low", "volume"]
    df.dropna(subset=feats, inplace=True)

    # auto-balance: only compute threshold once per (sym,horizon)
    if cfg.get("auto_balance", True) and "label_threshold" not in cfg:
        cfg["label_threshold"] = _auto_threshold(
            df, horizon=horizon, target_ratio=0.12, sym=sym
        )

    labels = (
        (df["high"].rolling(horizon).max().shift(-horizon) - df["mid"])
        >= cfg["label_threshold"]
    ).astype(int)
    df, labels = df.iloc[:-horizon], labels.iloc[:-horizon]

    seq_len = cfg["seq_len"]
    if len(df) < seq_len + horizon + 100:
        raise ValueError("Dataset too small")

    n = len(df)
    i_tr, i_val = int(n * 0.70), int(n * 0.85)

    scaler = StandardScaler().fit(df[feats].iloc[:i_tr].values)
    X = scaler.transform(df[feats].values)
    y = labels.to_numpy()

    return (
        (X[:i_tr], y[:i_tr]),
        (X[i_tr:i_val], y[i_tr:i_val]),
        (X[i_val:], y[i_val:]),
    ), scaler, feats

###############################################################################
# Context helper – cleans up temp checkpoint directories automatically
###############################################################################

@contextmanager
def _tmpdir(prefix: str):
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        # suppress errors if already deleted
        if path.exists():
            for p in path.glob("*"):
                p.unlink(missing_ok=True)
            path.rmdir()

###############################################################################
# Core training for a single hyper-param combo
###############################################################################

def _train_variant(cfg: dict, csv_path: Path) -> Tuple[Path, float, float]:
    # ----------------------------------------------------------- random seeds
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    seq_len = cfg["seq_len"]  # honour grid’s seq_len

    # --------------------------------------------------------------- prepare
    splits, scaler, feats = _prepare(pd.read_csv(csv_path), cfg)
    (Xtr, ytr), (Xval, yval), (Xtst, ytst) = splits

    for name, arr in [("val", yval), ("test", ytst)]:
        if (np.bincount(arr, minlength=2) == 0).any():
            raise ValueError(f"{name} split is missing a class")

    counts = np.bincount(ytr, minlength=2).astype(float)
    if (counts == 0).any():
        raise ValueError("training set missing a class")

    cls_w = torch.tensor(counts.sum() / (2 * counts), device=DEVICE, dtype=torch.float32)
    _LOG.info(
        "[%s] class counts 0:%d 1:%d → weights 0:%.4f 1:%.4f",
        cfg["symbol"], int(counts[0]), int(counts[1]), cls_w[0].item(), cls_w[1].item(),
    )

    # ------------------------------------------------ optional rebalancing
    if _needs_rebalance(ytr):
        weights = np.where(ytr == 1, counts[0] / max(counts[1], 1), 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True)
    else:
        sampler = None

    tr_loader = DataLoader(_SeqDS(Xtr, ytr, seq_len), batch_size=64, sampler=sampler, shuffle=(sampler is None))
    val_loader = DataLoader(_SeqDS(Xval, yval, seq_len), batch_size=64)
    dls = DataLoaders(tr_loader, val_loader, device=DEVICE)

    net = _LSTM(Xtr.shape[1], cfg["hidden"], cfg["dropout"])

    loss_fn = (
        FocalLoss(weight=cls_w) if cfg.get("focal", True) else torch.nn.CrossEntropyLoss(weight=cls_w)
    )
    f1 = F1Score(average="binary", pos_label=1)
    f1.name = "f1_score"

    def _pr_auc(preds, targs):
        probs_ = preds.softmax(dim=1)[:, 1].cpu().numpy()
        return average_precision_score(targs.cpu().numpy(), probs_)

    pr_auc_m = AccumMetric(_pr_auc, flatten=False, name="pr_auc")

    with _tmpdir(f"{cfg['symbol']}_ckpt_") as tmp_ckpt:
        learn = Learner(
            dls,
            net,
            loss_func=loss_fn,
            metrics=[accuracy, f1, RocAucBinary(), pr_auc_m, Precision(average="binary", pos_label=1), Recall(average="binary", pos_label=1)],
            model_dir=tmp_ckpt,
            cbs=[
                SaveBestCPU(monitor="f1_score", fname="best", with_opt=False),
                EarlyStoppingCallback(monitor="f1_score", patience=2),
                CSVLogger(fname=LOG_DIR / f"{cfg['symbol']}_train.csv"),
                StopFlagCallback(),
            ],
        )

        learn.fit_one_cycle(EPOCHS, LR)
        learn.load("best", with_opt=False, device=torch.device("cpu"))
        learn.model.cpu()

    # ----------------------------------- strip callbacks & make picklable
    learn.remove_cbs(learn.cbs)
    for dl in learn.dls.loaders:
        dl.sampler = None
        dl.shuffle = False
    learn.dls = learn.dls.new_empty()

    # -------------------------------------------------------------- test set
    tst_loader = DataLoader(_SeqDS(Xtst, ytst, seq_len), batch_size=128)
    logits, targets = [], []
    with torch.no_grad():
        for xb, yb in tst_loader:
            logits.append(learn.model(xb).cpu())
            targets.append(yb.cpu())
    logits = torch.cat(logits)
    targets = torch.cat(targets)

    preds = logits.argmax(1)
    probs = logits.softmax(1)[:, 1].numpy()
    f1_test = f1_score(targets.numpy(), preds.numpy(), average="binary", pos_label=1)
    pr_auc_t = average_precision_score(targets.numpy(), probs)
    _LOG.info("[%s] TEST-set F1 = %.4f (gate %.2f)", cfg["symbol"], f1_test, MIN_F1)

    # -------------------------------------------------------- threshold tune
    vt_loader = DataLoader(_SeqDS(Xval, yval, seq_len), batch_size=128)
    with torch.no_grad():
        v_probs = torch.cat([learn.model(xb).softmax(1)[:, 1].cpu() for xb, _ in vt_loader]).numpy()

    off = seq_len - 1
    prec, rec, thr = precision_recall_curve(yval[off:], v_probs)
    best_idx = int(np.argmax((2 * prec * rec) / (prec + rec + 1e-9)))
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    _LOG.info("Val threshold = %.4f (precision %.3f recall %.3f)", best_thr, prec[best_idx], rec[best_idx])

    # ------------------------------------------------------------ export paths
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bar   = cfg.get("interval_minutes", 1)        # ← defaults to 1 m if not set
    stem = (
        f"{cfg['symbol']}_b{bar}"                  # ← bar tag
        f"_h{cfg['horizon']}"
        f"_thr{cfg['label_threshold']}"
        f"_hid{cfg['hidden']}"
        f"_dp{cfg['dropout']}"
        f"_f1_{f1_test:.5f}"
        f"_{stamp}"
    )
    model_p  = MODELS_DIR / f"{stem}.pkl"
    scaler_p = MODELS_DIR / f"{stem}_sc.pkl"
    model_p.parent.mkdir(parents=True, exist_ok=True)

    learn.export(model_p)
    joblib.dump({"scaler": scaler, "features": feats, "threshold": best_thr}, scaler_p)
    _LOG.info("Model + scaler exported")

    # ------------------------------------------------------------------ 15-d BT
    try:
        stats = _quick_score_loud(cfg["symbol"], model_p)
        pnl15 = float(stats.get("pnl", np.nan))
    except Exception as exc:
        _LOG.warning("Back-test failed for %s: %s – meta will have NaN", model_p.name, exc)
        pnl15 = float("nan")

    meta = {
        "symbol":          cfg["symbol"],
        "bar":             bar,
        "horizon":         cfg["horizon"],
        "seq_len":         cfg["seq_len"],
        "label_threshold": cfg["label_threshold"],
        "f1_test":         f1_test,
        "pr_auc_test":     pr_auc_t,
        "pnl_15d":         pnl15,
        "ts":              datetime.now(timezone.utc).isoformat(),
    }
    model_p.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    return model_p, f1_test, pr_auc_t

###############################################################################
# Public convenience helpers  (NEW)
###############################################################################

def train_grid(
    cfg: dict,
    csv_path: Path,
    *,
    bars: Sequence[int] | None = None,
    preset: str = "scalp",
    depth: Literal["quick", "partial", "full"] = "full",
) -> list[Path]:
    """
    Train a hyper‑parameter grid (or subset) and return kept model paths.

    Parameters
    ----------
    cfg : dict
        Base coin/training config.
    csv_path : Path
        OHLC dataset to train on.
    bars : Sequence[int] | None
        List of bar sizes (minutes).  None → use cfg['interval_minutes'].
    preset, depth
        See docstring in original file.
    """
    kept: list[Path] = []
    bars = bars or [cfg.get("interval_minutes", 1)]
    bars = [b for b in bars if b in BARS]

    for bar in bars:
        cfg_bar = {**cfg, "interval_minutes": bar}
        grid_spec = build_hyper_grid(cfg_bar, preset=preset, bar=bar)
        keys = list(grid_spec)
        combos = list(product(*grid_spec.values()))

        if depth == "quick":
            combos = combos[:1]
        elif depth == "partial":
            seen_hs: set[tuple[int, int]] = set()
            uniq = []
            for cmb in combos:
                hz, sl = cmb[keys.index("horizon")], cmb[keys.index("seq_len")]
                if (hz, sl) not in seen_hs:
                    seen_hs.add((hz, sl))
                    uniq.append(cmb)
            combos = uniq

        res: list[tuple[float, Path]] = []
        for combo in combos:
            var_cfg = {**cfg_bar, **dict(zip(keys, combo))}
            try:
                mpath, f1_val, _ = _train_variant(var_cfg, csv_path)
                if f1_val >= MIN_F1:
                    res.append((f1_val, mpath))
            except (ValueError, CancelFitException):
                continue
            except Exception as exc:
                _LOG.warning("[%s] combo %s failed: %s", cfg_bar["symbol"], combo, exc)

        kept.extend(_select_top(res))

    return kept


def ensure_champion(
    cfg: dict,
    csv_path: Path,
    *,
    depth: Literal["quick", "partial", "full"] = "partial",
    preset: str = "scalp",
) -> Path:
    """
    Guarantee **one** champion model exists – return its path.
    Trains if necessary (uses `train_grid` under the hood).
    """
    champ = get_champion(cfg["symbol"])
    if champ and champ.exists():
        return champ

    # inject csv path so train_grid can reach _train_variant without signature change
    cfg_local = {**cfg, "_csv_path": csv_path}
    train_grid(cfg_local,csv_path, preset=preset, depth=depth)
    champ = get_champion(cfg["symbol"])
    if champ is None:
        raise RuntimeError(f"Champion still missing for {cfg['symbol']}")
    return champ

###############################################################################
# Public API: list_models, get_champion, get_next_candidate
###############################################################################

def list_models(symbol: str) -> List[Path]:
    return sorted(p for p in MODELS_DIR.glob(f"{symbol}_*.pkl") if not p.name.endswith("_sc.pkl"))

def get_champion(symbol: str) -> Path | None:
    models = list_models(symbol)
    return max(models, key=_extract_f1) if models else None

def get_next_candidate(symbol: str, *, exclude: set[Path] | None = None) -> Path | None:
    pool = [p for p in list_models(symbol) if p not in (exclude or set())]
    pool.sort(key=_extract_f1, reverse=True)
    return pool[0] if pool else None

###############################################################################
# High-level training loop (deprecated shim for train_and_select)
###############################################################################

def train_and_select(cfg: dict, csv_path: Path) -> List[Path]:
    """Train all hyper-param variants and return kept model paths."""
    combos = list(product(*HYPER_GRID.values()))
    if FAST_TRAIN:
        combos = combos[:1]

    start = time.time()
    results: List[tuple[float, Path]] = []

    for i, combo in enumerate(combos, 1):
        if stop_flag.is_set():
            break
        keys = list(HYPER_GRID)
        var_cfg = {**cfg, **dict(zip(keys, combo))}
        try:
            mpath, f1_t, pr_auc = _train_variant(var_cfg, csv_path)
            _LOG.info("%s → F1=%.4f", mpath.name, f1_t)
            if f1_t < MIN_F1:
                _LOG.info("%s below gate %.2f – discarded", mpath.name, MIN_F1)
                continue
            results.append((f1_t, mpath))

            # optional back-test (never discards on failure)
            try:
                stats = _quick_score_loud(cfg["symbol"], mpath)
            except Exception as exc:
                _LOG.warning("Back-test failed for %s: %s – model kept", mpath.name, exc)
                stats = {}
            pnl = stats.get("pnl", float("nan"))
            _LOG.info("%s  prAUC=%.3f  pnl=%+.6f  %d/%d", mpath.name, pr_auc, pnl, i, len(combos))

        except (ValueError, CancelFitException) as exc:
            _LOG.info("Variant %s skipped: %s", combo, exc)
        except Exception as exc:
            _LOG.warning("Variant %s failed: %s", combo, exc)

    # keep only top-N if configured
    if not KEEP_ALL_MODELS and len(results) > KEEP_TOP:
        results.sort(key=lambda t: t[0], reverse=True)
        for _, p in results[KEEP_TOP:]:
            p.unlink(missing_ok=True)
            p.with_name(p.stem + "_sc.pkl").unlink(missing_ok=True)
        results = results[:KEEP_TOP]

    return [p for _, p in results]
    # DEPRECATED: call train_grid instead
    # _LOG.warning("train_and_select() is deprecated – switch to train_grid()")
    # cfg = {**cfg, "_csv_path": csv_path}
    # return train_grid(cfg)
def parse_model_bar(model_path: Path) -> int:
    """
    Extracts the bar-size from filenames like SYMBOL_b15_…;
    returns 1 if no _b<digits>_ is found.
    """
    m = re.search(r"_b(\d+)", model_path.stem)
    return int(m.group(1)) if m else 1
__all__ = [

    "build_hyper_grid",
    "train_variant",
    "quick_backtest_15d",
    "serialize_model_scaler",
    "parse_model_bar",
]

def build_hyper_grid(
    cfg: dict = None,
    *,
    preset: str = "scalp",
    bar: int | None = None,
) -> Mapping[str, Sequence]:
    """
    External wrapper for trainer.py.

    Parameters
    ----------
    cfg : dict (1 minute default)
        Coin config (may contain 'interval_minutes').
    preset : str
        Grid preset.
    bar : int | None
        Bar length in minutes.  If omitted we derive it from cfg.
    """
    if cfg is None:
        cfg = {"interval_minutes": 1}
    return _build_hyper_grid(cfg, preset=preset, bar=bar)


def train_variant(cfg: dict, csv_path: Path):
    """
    Public façade for `_train_variant`.
    Returns (model_path, f1_test, pr_auc_test) exactly like the private one.
    """
    return _train_variant(cfg, csv_path)


def serialize_model_scaler(model, scaler, model_path: Path, scaler_path: Path) -> None:
    """
    Tiny helper used by trainer.py – dumps model + scaler side‑by‑side.
    (Feel free to extend with metadata if you wish.)
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu(), model_path)
    joblib.dump(scaler, scaler_path)


def quick_backtest_15d(model, scaler, df: pd.DataFrame, horizon: int) -> float:
    """
    Minimal 15‑day back‑tester for trainer.py’s keep/discard gate.

    Strategy (very simple, replace with your own):
      • feed the LAST 15 days of data through `model`
      • go LONG 1 unit whenever P(pos) > 0.5, FLAT otherwise
      • compute cumulative mid‑price return

    Returns the gross PnL in *price points* (not %), so trainer just checks
    `pnl > 0` to decide whether to keep the variant.
    """
    if "mid" not in df.columns:
        raise ValueError("DF must contain a 'mid' column")

    # figure out how many rows ≈ 15 days for this bar‑size
    minutes = df.attrs.get("interval_minutes", 1)
    rows_needed = int(15 * 24 * 60 / minutes)
    tail = df.tail(rows_needed).copy()
    if len(tail) < rows_needed:
        return float("nan")

    feats = ["mid", "high", "low", "volume"]
    X = scaler.transform(tail[feats].values)
    seq_len = getattr(model, "seq_len", DEFAULT_SEQ_LEN)  # best‑effort
    ds = _SeqDS(X, np.zeros(len(X), dtype=int), seq_len)
    dl = DataLoader(ds, batch_size=256)

    import itertools, numpy as np
    probs = []
    with torch.no_grad():
        for xb, _ in dl:
            probs.append(model(xb).softmax(1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)

    # Generate positions (long‑only, threshold 0.5)
    pos = probs > 0.5
    mid = tail["mid"].values[seq_len - 1 :]  # align with label index
    returns = np.diff(mid) / mid[:-1]
    pnl = float(np.sum(returns * pos[:-1]))  # ignore last step (no next‑price)

    return pnl