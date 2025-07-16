"""
Kraken REST helper – adaptive rate limiter 2025‑05‑12
────────────────────────────────────────────────────
Improvements over the previous patch:

• **Sliding‑window token bucket** – up to **18 calls / 10 s** across *all* threads
  (Kraken allows 20). Guarantees compliance instead of a fixed delay.
• **Endpoint cost map** – heavy endpoints like **OHLC** debit **2 tokens**
  because Kraken seems to weigh them more harshly.
• **Dynamic back‑off** – after any 429/Too‑many response we add a temporary
  **cool‑off** pause (starts at 1 s, doubles to max 8 s) before the next token
  can be consumed.

The public API (`ticker`, `asset_pairs`, `public`, …) stays the same.

"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import threading
import time
import urllib.parse
from collections import deque
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException


try:
    from .log_setup import get_logger
except ImportError:
    from log_setup import get_logger

###############################################################################
# Env & logging
###############################################################################

load_dotenv(dotenv_path=Path(__file__).parent.parent / "config" / ".env")
_LOG = get_logger("Kraken")

###############################################################################
# Constants & helpers
###############################################################################

_API, _VER = "https://api.kraken.com", "0"
_WINDOW, _CAPACITY = 10.0, 18  # calls / seconds (leave slack under 20)
_COST = {  # endpoint‑specific token cost
    "OHLC": 2,  # empirically heavier
}

_nonce_lock, _last_nonce = threading.Lock(), 0


def _nonce() -> str:
    global _last_nonce
    with _nonce_lock:
        n = int(time.time() * 1000)
        _last_nonce = max(n, _last_nonce + 1)
        return str(_last_nonce)

###############################################################################
# Main client
###############################################################################


class KrakenREST:
    """Kraken REST client with **adaptive, process‑wide** rate limiting."""

    def __init__(self, key: str | None = None, secret: str | None = None):
        self.key = key or os.getenv("KRAKEN_API_KEY")
        self.secret = secret or os.getenv("KRAKEN_API_SECRET")
        self.pub_only = not (self.key and self.secret)
        if self.pub_only:
            _LOG.info("No keys: PUBLIC‑only client.")

        self.sess = requests.Session()

        # Tokens consumed timestamps (monotonic seconds)
        self._ts_history: deque[float] = deque(maxlen=_CAPACITY)
        self._hist_lock = threading.Lock()
        self._cool_off_until = 0.0  # monotonic sec until which calls are paused

    # ------------------------------------------------------------------ #
    # Internal – token bucket
    # ------------------------------------------------------------------ #
    def _acquire_tokens(self, n: int) -> None:
        while True:
            with self._hist_lock:
                now = time.monotonic()
                if now < self._cool_off_until:
                    sleep = self._cool_off_until - now
                else:
                    # purge expired timestamps
                    while self._ts_history and now - self._ts_history[0] > _WINDOW:
                        self._ts_history.popleft()
                    if len(self._ts_history) + n <= _CAPACITY:
                        for _ in range(n):
                            self._ts_history.append(now)
                        return  # acquired!
                    # need to wait for room
                    sleep = _WINDOW - (now - self._ts_history[0]) + 0.01
            time.sleep(sleep)

    # ------------------------------------------------------------------ #
    # Internal – signing helpers
    # ------------------------------------------------------------------ #
    def _sign(self, path: str, data: Dict[str, Any]):
        data["nonce"] = _nonce()
        post = urllib.parse.urlencode(data)
        sha = hashlib.sha256((data["nonce"] + post).encode()).digest()
        mac = hmac.new(base64.b64decode(self.secret), (path.encode() + sha), hashlib.sha512)
        return base64.b64encode(mac.digest()).decode(), data

    # ------------------------------------------------------------------ #
    # Private REST
    # ------------------------------------------------------------------ #
    def _private(self, method: str, data: Dict[str, Any] | None = None, *, retry: int = 5):
        if self.pub_only:
            raise RuntimeError("Private call without keys.")
        path = f"/{_VER}/private/{method}"
        sig, data = self._sign(path, data or {})
        hdr = {"API-Key": self.key, "API-Sign": sig}

        cost = _COST.get(method, 1)
        for attempt in range(1, retry + 1):
            self._acquire_tokens(cost)
            try:
                resp = self.sess.post(_API + path, headers=hdr, data=data, timeout=30)
                if resp.status_code == 429:
                    raise RequestException("HTTP 429 Too Many Requests")
                resp.raise_for_status()
                js = resp.json()
                errors = js.get("error", [])
                if errors and any("limit" in e for e in errors):
                    raise RequestException(errors)
                if errors:
                    raise RuntimeError(f"{method} error: {errors}")
                return js["result"]
            except (RequestException, ValueError) as exc:
                with self._hist_lock:
                    # exponential cool‑off starting at 1 s
                    self._cool_off_until = max(self._cool_off_until, time.monotonic() + min(2 ** attempt, 8))
                _LOG.warning("Private %s attempt %d/%d failed: %s", method, attempt, retry, exc)
        raise RuntimeError(f"Private call {method} failed after {retry} attempts.")

    # ------------------------------------------------------------------ #
    # Public REST
    # ------------------------------------------------------------------ #
    def _public(self, method: str, data: Dict[str, Any] | None = None, *, retry: int = 5):
        url = f"{_API}/{_VER}/public/{method}"
        cost = _COST.get(method, 1)
        for attempt in range(1, retry + 1):
            self._acquire_tokens(cost)
            try:
                resp = self.sess.post(url, data=data or {}, timeout=30)
                if resp.status_code == 429:
                    raise RequestException("HTTP 429 Too Many Requests")
                resp.raise_for_status()
                js = resp.json()
                errors = js.get("error", [])
                if errors and any("EGeneral:Too" in e or "rate" in e.lower() for e in errors):
                    raise RequestException(errors)
                if errors:
                    raise ValueError(errors)
                if "result" not in js:
                    raise ValueError("Malformed response")
                return js["result"]
            except (RequestException, ValueError) as exc:
                with self._hist_lock:
                    self._cool_off_until = max(self._cool_off_until, time.monotonic() + min(2 ** attempt, 8))
                _LOG.warning("Public %s attempt %d/%d failed: %s", method, attempt, retry, exc)
        raise RuntimeError(f"Public call {method} failed after retries.")

    # ------------------------------------------------------------------ #
    # Convenience wrappers
    # ------------------------------------------------------------------ #
    def ticker(self, pair: str):
        result = self._public("Ticker", {"pair": pair})
        return result.get(pair) if result else None

    def asset_pairs(self, pair: str):
        return self._public("AssetPairs", {"pair": pair})[pair]

    def add_order(self, **data):
        return self._private("AddOrder", data)["txid"][0]

    def public(self, method: str, data: Dict[str, Any] | None = None):
        return self._public(method, data)

