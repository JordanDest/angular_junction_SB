# """
# core/ktrader.py
# ────────────────
# Stateless(ish) Kelly‑style trader re‑implementation.

# * Renamed from KellyTrader  →  KTrader
# * Uses utils.safe_convert, not data_validation.safe_convert
# * All internal magic numbers pulled into .cfg so Tournament can override
# """
# from __future__ import annotations

# import warnings
# from collections import deque
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Deque, List

# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from fastai.learner import load_learner

# from utils.utils import safe_convert


# class KTrader:
#     SEQ_LEN = 60

#     def __init__(
#         self,
#         coin_cfg: dict,
#         model_path: Path,
#         global_equity: List[float] | None = None,
#         starting_capital: float = 100.0,
#     ):
#         self.cfg = {
#             "slippage_bps": 0.0,
#             "fee_bps": 0.0,
#             "bull_bias_bps": 0.0,
#             "kelly_fraction": 1.0,
#             "entry_threshold": 0.50,
#             "trail_stop_pct": 0.01,
#             "take_profit_pct": 0.02,
#             "stop_loss_atr_mult": 2.0,
#             "break_even_pct": 0.005,
#             "atr_period": 14,
#             **coin_cfg,
#         }

#         self.capital = starting_capital
#         self.cash = starting_capital
#         self.global_eq = global_equity if global_equity else [starting_capital]
#         self.trade_log: list[dict] = []

#         self._reset_position()

#         model_path = Path(model_path)
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", message="load_learner` uses Python's insecure pickle module", category=UserWarning)
#             self.learner = load_learner(model_path, cpu=True)
#         self.learner.model.eval()

#         self.model_name = model_path.name
#         scaler_data = joblib.load(model_path.with_name(f"{model_path.stem}_sc.pkl"))
#         self.scaler = scaler_data["scaler"]
#         self.features: list[str] = scaler_data["features"]

#         self.feature_buffer: Deque[list[float]] = deque(maxlen=self.SEQ_LEN)
#         self.atr_buffer: Deque[float] = deque(maxlen=self.cfg["atr_period"])

#     def _reset_position(self) -> None:
#         self.entry_price = 0.0
#         self.pos_units = 0.0
#         self.max_price = 0.0
#         self.stake_usd = 0.0

#     def _save_log(self):
#         if self.trade_log:
#             df = pd.DataFrame(self.trade_log)
#             file = Path(f"trade_log_{self.cfg['symbol']}.csv")
#             file_exists = file.exists()
#             df.to_csv(file, mode='a', header=not file_exists, index=False)
#             self.trade_log.clear()

#     def _enter(self, units: float, price: float, prob_up: float, kelly: float, stake_usd: float) -> None:
#         self.entry_price = price
#         self.pos_units = units
#         self.stake_usd = stake_usd
#         self.cash -= self.stake_usd
#         self.max_price = price

#         self.trade_log.append({
#             "ts": datetime.now(timezone.utc).isoformat(),
#             "symbol": self.cfg["symbol"],
#             "side": "open",
#             "qty": units,
#             "price": price,
#             "mode": "paper",
#             "cap_before": self.capital,
#             "cap_after": self.capital,
#             "pnl": 0.0,
#             "prob_up": prob_up,
#             "kelly_frac": kelly,
#             "stake_usd": stake_usd,
#             "reason": "conf_entry",
#             "model": self.model_name,
#         })
#         self._save_log()

#     def _exit(self, price: float, reason: str = "unknown") -> None:
#         pnl = self.pos_units * (price - self.entry_price)
#         cap_before = self.capital
#         self.cash += self.stake_usd + pnl
#         self.capital = self.cash
#         self.global_eq[0] += self.capital - cap_before

#         self.trade_log.append({
#             "ts": datetime.now(timezone.utc).isoformat(),
#             "symbol": self.cfg["symbol"],
#             "side": "close",
#             "qty": self.pos_units,
#             "price": price,
#             "mode": "paper",
#             "cap_before": cap_before,
#             "cap_after": self.capital,
#             "pnl": pnl,
#             "prob_up": None,
#             "kelly_frac": None,
#             "stake_usd": self.stake_usd,
#             "reason": reason,
#             "model": self.model_name,
#         })
#         self._save_log()
#         self._reset_position()

#     def _kelly_fraction(self, p: float) -> float:
#         q = 1.0 - p
#         variance = p * q
#         if variance == 0.0:
#             return 0.0

#         raw = (p - q) / variance
#         scaled = self.cfg["kelly_fraction"] * raw
#         return max(0.0, min(scaled, 0.25))

#     def _predict_up_probability(self, row: pd.Series) -> float:
#         features_row = [safe_convert(row[f], float, None) for f in self.features]
#         if any(v is None for v in features_row) or not np.isfinite(features_row).all():
#             return 0.5

#         self.feature_buffer.append(features_row)
#         if len(self.feature_buffer) < self.SEQ_LEN:
#             return 0.5

#         seq = np.array(self.feature_buffer, dtype=np.float32)
#         seq = self.scaler.transform(seq)
#         tensor_inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
#         tensor_inp = tensor_inp.to(next(self.learner.model.parameters()).device)

#         with torch.no_grad():
#             logits = self.learner.model(tensor_inp)
#             probs = torch.softmax(logits, dim=1)
#         return float(probs[0, 1].cpu())
    
#     def _target_units(self, price: float, prob_up: float) -> float:
#         """Dynamic position size based on equity, confidence, and volatility."""
#         atr   = np.mean(self.atr_vals) if self.atr_vals else price * 0.01
#         edge  = prob_up - 0.5
#         var   = prob_up * (1 - prob_up) + 1e-9
#         kelly = edge / var
#         conf  = np.clip(0.5 + self.cfg["kelly_fraction"] * kelly, 0.0, 2.0)

#         dollars_at_risk = self.cfg["risk_frac_base"] * self.global_eq[0] * conf
#         unit_risk       = atr * self.cfg["stop_loss_atr_mult"]
#         return dollars_at_risk / unit_risk


#     def _compute_atr(self) -> float:
#         if len(self.atr_buffer) < self.cfg["atr_period"]:
#             return 0.0
#         return float(np.mean(self.atr_buffer))

#     def step(self, raw_row: dict | pd.Series) -> None:
#         row = pd.Series(raw_row) if isinstance(raw_row, dict) else raw_row.copy()
#         for fld in ("high", "low", "volume"):
#             row[fld] = safe_convert(row.get(fld), float, np.nan)
#         row["mid"] = safe_convert(row.get("mid"), float, (row["high"] + row["low"]) / 2)
#         row["spread"] = safe_convert(row.get("spread"), float, row["high"] - row["low"])

#         self.atr_buffer.append(row["high"] - row["low"])
#         prob_up = self._predict_up_probability(row)

#         if self.pos_units > 0:
#             mid = row["mid"]
#             self.max_price = max(self.max_price, mid)
#             atr = self._compute_atr()
#             sl_atr = self.entry_price - self.cfg["stop_loss_atr_mult"] * atr
#             if mid >= self.entry_price * (1 + self.cfg["break_even_pct"]):
#                 sl_atr = max(sl_atr, self.entry_price)
#             trail_sl = self.max_price * (1 - self.cfg["trail_stop_pct"])
#             take_profit = self.entry_price * (1 + self.cfg["take_profit_pct"])

#             if mid <= min(sl_atr, trail_sl):
#                 self._exit(mid, reason="ATR_SL" if mid <= sl_atr else "trail_SL")
#             elif mid >= take_profit:
#                 self._exit(mid, reason="TP")
#             return

#         if prob_up >= self.cfg["entry_threshold"]:
#             kelly = self._kelly_fraction(prob_up)
#             eq = self.global_eq[0]
#             risk_budget = (0.05 if eq < 250 else 0.02 if eq < 1_000 else 0.01) * eq
#             conf_bonus = 2.0 * (prob_up - self.cfg["entry_threshold"])
#             stake_usd = risk_budget * (kelly + conf_bonus)
#             stake_usd = max(1.0, min(stake_usd, risk_budget))

#             if stake_usd >= 1.0:
#                 units = stake_usd / row["mid"]
#                 self._enter(units, row["mid"], prob_up, kelly, stake_usd)

#     def liquidate(self, price: float | None = None) -> None:
#         if self.pos_units <= 0:
#             return

#         exit_price = price if price is not None else self.entry_price
#         self._exit(exit_price, reason="liquidate")



"""
core/ktrader.py
───────────────
Position‑sizing, Kelly‑aware trader.

Changes vs previous version
• Pull every sizing/risk constant from cfg so the sizing‑grid search can override.
• Introduce _target_units():  position size ∝ equity × confidence / volatility.
• Replace old stake_usd block in step() with new _target_units() call.
• Adds default cfg["risk_frac_base"] so trading.py override is optional.
• Naming clean‑up: uses self.atr_buffer throughout.
"""
from __future__ import annotations

import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List

import joblib
import numpy as np
import pandas as pd
import torch
from fastai.learner import load_learner

from utils.utils import safe_convert


class KTrader:
    """Stateless(ish) trader that sizes positions by Kelly‑scaled risk.

    A single instance manages ONE symbol.  Capital auto‑compounds via the
    shared *global_equity* list provided by pipelines/trading.py.
    """

    SEQ_LEN = 60  # model expects 60 timesteps

    def __init__(
        self,
        coin_cfg: dict,
        model_path: Path,
        global_equity: List[float] | None = None,
        starting_capital: float = 100.0,
    ) -> None:
        # ───── config ──────────────────────────────────────────────
        self.cfg = {
            # execution‑related
            "slippage_bps": 0.0,
            "fee_bps": 0.0,
            "bull_bias_bps": 0.0,
            # sizing constants (override by SIZE_GRID)
            "risk_frac_base": 0.0025,          # %‑equity at p=0.5, conf=1.0
            "kelly_fraction": 1.0,             # multiplies Kelly edge
            "stop_loss_atr_mult": 2.0,
            # entry/exit logic
            "entry_threshold": 0.50,
            "trail_stop_pct": 0.01,
            "take_profit_pct": 0.02,
            "break_even_pct": 0.005,
            "atr_period": 14,
            # user overrides
            **coin_cfg,
        }

        # ───── capital state ───────────────────────────────────────
        self.capital = starting_capital
        self.cash = starting_capital
        self.global_eq = global_equity if global_equity else [starting_capital]
        self.trade_log: list[dict] = []

        self._reset_position()

        # ───── model ------------------------------------------------
        model_path = Path(model_path)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="load_learner` uses Python's insecure pickle module",
                category=UserWarning,
            )
            self.learner = load_learner(model_path, cpu=True)
        self.learner.model.eval()
        self.model_name = model_path.name

        scaler_data = joblib.load(model_path.with_name(f"{model_path.stem}_sc.pkl"))
        self.scaler = scaler_data["scaler"]
        self.features: list[str] = scaler_data["features"]

        # ───── rolling buffers -------------------------------------
        self.feature_buffer: Deque[list[float]] = deque(maxlen=self.SEQ_LEN)
        self.atr_buffer: Deque[float] = deque(maxlen=self.cfg["atr_period"])

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────
    def _reset_position(self) -> None:
        self.entry_price = 0.0
        self.pos_units = 0.0
        self.max_price = 0.0
        self.stake_usd = 0.0

    # -------------------------------------------------------------------
    # Logging helpers (unchanged)
    # -------------------------------------------------------------------
    def _save_log(self) -> None:
        if not self.trade_log:
            return
        df = pd.DataFrame(self.trade_log)
        file = Path(f"trade_log_{self.cfg['symbol']}.csv")
        header = not file.exists()
        df.to_csv(file, mode="a", header=header, index=False)
        self.trade_log.clear()

    def _enter(self, units: float, price: float, prob_up: float, kelly: float, stake_usd: float) -> None:
        self.entry_price = price
        self.pos_units = units
        self.stake_usd = stake_usd
        self.cash -= self.stake_usd
        self.max_price = price

        self.trade_log.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": self.cfg["symbol"],
                "side": "open",
                "qty": units,
                "price": price,
                "mode": "paper",
                "cap_before": self.capital,
                "cap_after": self.capital,
                "pnl": 0.0,
                "prob_up": prob_up,
                "kelly_frac": kelly,
                "stake_usd": stake_usd,
                "reason": "conf_entry",
                "model": self.model_name,
            }
        )
        self._save_log()

    def _exit(self, price: float, reason: str = "unknown") -> None:
        pnl = self.pos_units * (price - self.entry_price)
        cap_before = self.capital
        self.cash += self.stake_usd + pnl
        self.capital = self.cash
        self.global_eq[0] += self.capital - cap_before

        self.trade_log.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": self.cfg["symbol"],
                "side": "close",
                "qty": self.pos_units,
                "price": price,
                "mode": "paper",
                "cap_before": cap_before,
                "cap_after": self.capital,
                "pnl": pnl,
                "prob_up": None,
                "kelly_frac": None,
                "stake_usd": self.stake_usd,
                "reason": reason,
                "model": self.model_name,
            }
        )
        self._save_log()
        self._reset_position()

    # -------------------------------------------------------------------
    # Prediction + sizing
    # -------------------------------------------------------------------
    def _kelly_fraction(self, p: float) -> float:
        """Return fractional Kelly bet sizing scalar in [0, 0.25]."""
        q = 1.0 - p
        var = p * q
        if var == 0.0:
            return 0.0
        raw = (p - q) / var  # classic Kelly
        scaled = self.cfg["kelly_fraction"] * raw
        return max(0.0, min(scaled, 0.25))

    def _predict_up_probability(self, row: pd.Series) -> float:
        features_row = [safe_convert(row[f], float, None) for f in self.features]
        if any(v is None for v in features_row) or not np.isfinite(features_row).all():
            return 0.5

        self.feature_buffer.append(features_row)
        if len(self.feature_buffer) < self.SEQ_LEN:
            return 0.5

        seq = np.array(self.feature_buffer, dtype=np.float32)
        seq = self.scaler.transform(seq)
        tensor_inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        tensor_inp = tensor_inp.to(next(self.learner.model.parameters()).device)

        with torch.no_grad():
            logits = self.learner.model(tensor_inp)
            probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].cpu())

    def _compute_atr(self) -> float:
        if len(self.atr_buffer) < self.cfg["atr_period"]:
            return 0.0
        return float(np.mean(self.atr_buffer))

    def _target_units(self, price: float, prob_up: float) -> float:
        """Number of units that puts *risk_frac_base × equity × conf* at risk."""
        atr = self._compute_atr() or price * 0.01  # fallback 1 % of price
        edge = prob_up - 0.5
        var = prob_up * (1 - prob_up) + 1e-9
        kelly = edge / var
        conf = np.clip(0.5 + self.cfg["kelly_fraction"] * kelly, 0.0, 2.0)

        dollars_at_risk = self.cfg["risk_frac_base"] * self.global_eq[0] * conf
        unit_risk = atr * self.cfg["stop_loss_atr_mult"]
        return max(0.0, dollars_at_risk / unit_risk)

    # -------------------------------------------------------------------
    # Main driver per candle
    # -------------------------------------------------------------------
    def step(self, raw_row: dict | pd.Series) -> None:
        row = pd.Series(raw_row) if isinstance(raw_row, dict) else raw_row.copy()
        for fld in ("high", "low", "volume"):
            row[fld] = safe_convert(row.get(fld), float, np.nan)
        row["mid"] = safe_convert(row.get("mid"), float, (row["high"] + row["low"]) / 2)
        row["spread"] = safe_convert(row.get("spread"), float, row["high"] - row["low"])

        # update ATR
        self.atr_buffer.append(row["high"] - row["low"])

        prob_up = self._predict_up_probability(row)
        mid = row["mid"]

        # ─── manage open position ────────────────────────────────────
        if self.pos_units > 0:
            self.max_price = max(self.max_price, mid)
            atr = self._compute_atr()
            sl_atr = self.entry_price - self.cfg["stop_loss_atr_mult"] * atr
            if mid >= self.entry_price * (1 + self.cfg["break_even_pct"]):
                sl_atr = max(sl_atr, self.entry_price)
            trail_sl = self.max_price * (1 - self.cfg["trail_stop_pct"])
            take_profit = self.entry_price * (1 + self.cfg["take_profit_pct"])

            if mid <= min(sl_atr, trail_sl):
                self._exit(mid, reason="SL_atr" if mid <= sl_atr else "SL_trail")
            elif mid >= take_profit:
                self._exit(mid, reason="TP")
            return  # processed open position

        # ─── evaluate new entry ─────────────────────────────────────
        if prob_up < self.cfg["entry_threshold"]:
            return  # no long entry (shorts not implemented yet)

        units = self._target_units(price=mid, prob_up=prob_up)
        stake_usd = units * mid
        if stake_usd < 1.0:
            return  # ignore dust entries

        kelly = self._kelly_fraction(prob_up)
        self._enter(units, mid, prob_up, kelly, stake_usd)

    # -------------------------------------------------------------------
    def liquidate(self, price: float | None = None) -> None:
        if self.pos_units <= 0:
            return
        exit_price = price if price is not None else self.entry_price
        self._exit(exit_price, reason="liquidate")
