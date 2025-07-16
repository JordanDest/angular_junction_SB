"""utils/orch_utils.py – common plumbing for the coin‑autotrader stack
====================================================================
A *single* well‑tested utility module that hides the mundane details of
logging, configuration, data hygiene, persistence and fault‑tolerance so that
`tasks/*`, `orchestration/*` and notebooks can focus on business logic.

The API surface is intentionally small & opinionated – revisit when your
use‑case diverges.

-------------------------------------------------------------------------------
TL;DR of what’s inside
-------------------------------------------------------------------------------
* **Logging** – JSON or plain, stdout + rotating file, plus `timeit` decorator.
* **Config**  – YAML loader with env‑var overrides (`FOO__BAR` ⇒ `foo.bar`).
* **Filesystem helpers** – `ensure_dir`, `atomic_write`, `gen_run_id`.
* **Execution wrappers** – `retry` & `safe_run`.
* **Pandas hygiene** – `sanitize_candles`, `assert_frame_ok`.
* **Model persistence** – `save_model`, `load_model`, thin pickle wrappers.
* **Capital Ledger** – atomic JSON file with balance + lock.

-------------------------------------------------------------------------------
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import pickle
import random
import string
import sys
import textwrap
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Mapping

import yaml

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dep for non‑ML tasks
    pd = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_JSON_FMT = {
    "time": "%(asctime)s",
    "level": "%(levelname)s",
    "name": "%(name)s",
    "msg": "%(message)s",
}
_PLAIN_FMT = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _build_formatter(json_fmt: bool) -> logging.Formatter:  # noqa: D401
    """Return either a JSON or plain formatter."""
    if json_fmt:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:  # noqa: D401
                d = _JSON_FMT.copy()
                d["time"] = self.formatTime(record, _DATE_FMT)
                d["level"] = record.levelname
                d["name"] = record.name
                d["msg"] = record.getMessage()
                return json.dumps(d, ensure_ascii=False)
        return _JsonFormatter()
    return logging.Formatter(_PLAIN_FMT, datefmt=_DATE_FMT)


def configure_logging(
    *,
    level: int = logging.INFO,
    log_dir: str | Path | None = "logs",
    json_format: bool = False,
) -> logging.Logger:
    """Init root logger and return it.  *Idempotent* – safe to call many times."""
    root = logging.getLogger()
    if getattr(root, "_orch_utils_init", False):  # already configured
        return root

    # stdout handler ---------------------------------------------------------
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(_build_formatter(json_format))

    handlers: list[logging.Handler] = [stdout_handler]

    # file handler -----------------------------------------------------------
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "orchestrator.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(_build_formatter(json_format))
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    root._orch_utils_init = True  # type: ignore[attr-defined]
    return root


# Singleton default logger ----------------------------------------------------
log = configure_logging()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env_to_dict(prefix: str | None = None) -> dict[str, Any]:
    """Translate ENV_VARS into a dotted‑key dict.  E.g. FOO__BAR=1 → {foo: {bar:1}}"""
    out: dict[str, Any] = {}
    for k, v in os.environ.items():
        if prefix and not k.startswith(prefix):
            continue
        key = k[len(prefix):] if prefix else k
        parts = key.lower().split("__")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})  # type: ignore[assignment]
        cur[parts[-1]] = v
    return out


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:  # noqa: D401
    """Deep merge b into a, returning *new* dict."""
    out = json.loads(json.dumps(a))  # simple deepcopy
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_config(
    cfg_path: str | Path = "config/settings.yaml",
    *,
    env_prefix: str | None = None,
) -> dict[str, Any]:
    """Load YAML config and merge any environment overrides."""
    cfg_path = Path(cfg_path)
    base: dict[str, Any] = {}
    if cfg_path.exists():
        base = yaml.safe_load(cfg_path.read_text()) or {}
    env_patch = _env_to_dict(env_prefix)
    return _merge(base, env_patch)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def gen_run_id(ts: datetime | None = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    rand = "".join(random.choices(string.hexdigits.lower(), k=5))
    return ts.strftime("%Y%m%dT%H%M%S") + "_" + rand


@contextlib.contextmanager
def atomic_write(path: str | Path, mode: str = "w", encoding: str | None = "utf-8"):
    """Write to *path* atomically (temp‑file swap)."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open(mode, encoding=encoding) as fh:
            yield fh
        tmp.replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def retry(
    exceptions: tuple[type[Exception], ...] = (Exception,),
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
):
    """Decorator for exponential‑backoff retries."""
    def decorator(fn: Callable[..., Any]):
        @functools.wraps(fn)
        def wrapper(*args: Any, **kw: Any):  # noqa: D401 – nested function
            _delay = delay
            for attempt in range(retries):
                try:
                    return fn(*args, **kw)
                except exceptions as exc:
                    if attempt == retries - 1:
                        raise
                    log.warning("%s failed (%s); retrying in %.1fs", fn.__name__, exc, _delay)
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator


def timeit(fn: Callable[..., Any]):
    """Decorator that logs execution time (DEBUG level)."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kw: Any):  # noqa: D401
        start = time.perf_counter()
        res = fn(*args, **kw)
        log.debug("%s finished in %.2f s", fn.__name__, time.perf_counter() - start)
        return res
    return wrapper


def safe_run(
    logger: logging.Logger | None,
    fn: Callable[..., Any],
    *args: Any,
    retries: int = 0,
    **kw: Any,
) -> Any | None:  # noqa: ANN401 – allow Any
    """Run *fn* synchronously; swallow & log exceptions.

    Returns the function’s return value or *None* if it raised.
    """
    _log = logger or log
    try:
        _fn = fn
        if retries:
            _fn = retry(retries=retries)(fn)
        return _fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        _log.exception("safe_run: %s failed – %s", fn.__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Pandas helpers
# ---------------------------------------------------------------------------

if pd is not None:

    def sanitize_candles(df: "pd.DataFrame", freq: str = "1min") -> "pd.DataFrame":
        """Drop dupes, sort index, forward‑fill gaps to a regular frequency."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must be indexed by DatetimeIndex")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        df = df.reindex(full_idx)
        df.ffill(inplace=True)
        return df

    def assert_frame_ok(df: "pd.DataFrame", cols: Iterable[str] | None = None) -> None:
        """Simple sanity checks – no NaNs, monotonic index, required cols present."""
        if df.isna().any().any():
            raise ValueError("DataFrame contains NaNs after sanitization")
        if not df.index.is_monotonic_increasing:
            raise ValueError("Index not sorted")
        if cols and set(cols) - set(df.columns):
            raise ValueError(f"Missing columns: {set(cols) - set(df.columns)}")


# ---------------------------------------------------------------------------
# Model & pickle persistence helpers
# ---------------------------------------------------------------------------

PICKLE_PROTO = pickle.HIGHEST_PROTOCOL


def save_pickle(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with atomic_write(path, "wb", encoding=None) as fh:  # type: ignore[arg-type]
        pickle.dump(obj, fh, protocol=PICKLE_PROTO)
    return path


def load_pickle(path: str | Path) -> Any:  # noqa: ANN401 – allow Any
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def save_model(model: Any, meta: Mapping[str, Any], dir_: str | Path) -> Path:  # noqa: ANN401 – allow Any
    dir_ = ensure_dir(dir_)
    run_id = gen_run_id()
    model_path = Path(dir_) / f"model_{run_id}.pkl"
    meta_path = model_path.with_suffix(".json")
    save_pickle(model, model_path)
    with atomic_write(meta_path) as fh:
        json.dump(meta, fh, indent=2)
    log.info("Model saved: %s", model_path.name)
    return model_path


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:  # noqa: ANN401 – allow Any
    path = Path(path)
    model = load_pickle(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return model, meta


# ---------------------------------------------------------------------------
# Capital ledger (tiny, atomic JSON DB)
# ---------------------------------------------------------------------------

class Ledger:
    """Thread‑safe JSON ledger with *balance* & *locked* funds."""

    _lock = Lock()

    def __init__(self, path: str | Path = "state/ledger.json") -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        if not self.path.exists():
            self._write({"free": 0.0, "locked": 0.0})

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _read(self) -> dict[str, float]:
        return json.loads(self.path.read_text())

    def _write(self, data: dict[str, float]) -> None:
        with atomic_write(self.path) as fh:
            json.dump(data, fh)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @property
    def balance(self) -> float:
        with self._lock:
            d = self._read()
            return d["free" ] + d["locked"]

    @property
    def free(self) -> float:
        with self._lock:
            return self._read()["free"]

    def credit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        with self._lock:
            d = self._read()
            d["free"] += amount
            self._write(d)

    def lock(self, amount: float) -> bool:
        """Lock *amount* if enough free funds exist.  Returns *True* on success."""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        with self._lock:
            d = self._read()
            if d["free"] < amount:
                return False
            d["free"] -= amount
            d["locked"] += amount
            self._write(d)
            return True

    def release(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        with self._lock:
            d = self._read()
            if d["locked"] < amount:
                raise ValueError("Not enough locked funds")
            d["locked"] -= amount
            d["free"] += amount
            self._write(d)


# ---------------------------------------------------------------------------
# Convenience re‑exports
# ---------------------------------------------------------------------------

__all__ = [
    "log",
    "configure_logging",
    "load_config",
    "ensure_dir",
    "gen_run_id",
    "atomic_write",
    "retry",
    "timeit",
    "safe_run",
    "sanitize_candles" if pd is not None else "",
    "assert_frame_ok" if pd is not None else "",
    "save_model",
    "load_model",
    "save_pickle",
    "load_pickle",
    "Ledger",
]
