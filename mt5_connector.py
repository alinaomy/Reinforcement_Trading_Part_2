"""
MT5 Connector for PPO XAUUSD H1 Bracket Trading Model
======================================================
Runs on every completed H1 bar, computes features, queries the PPO model,
and sends bracket orders (SL + TP set at entry) to MetaTrader 5.

Requirements:
    pip install MetaTrader5 stable-baselines3 gymnasium pandas numpy python-dotenv

Usage:
    python mt5_connector.py                  # live trading (real account)
    python mt5_connector.py --dry-run        # compute signals but send no orders
    python mt5_connector.py --login 12345 --password "xxx" --server "Broker-Demo"
"""
from __future__ import annotations

import argparse
import logging
import os
import time

# Load .env file automatically so credentials don't need to be exported manually
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to shell-exported env vars
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mt5_connector")

# ── Project imports ────────────────────────────────────────────────────────────
from config import CFG
from features import add_stationary_features

# ── Constants ─────────────────────────────────────────────────────────────────
SYMBOL       = "XAUUSD"
TIMEFRAME_H1 = None          # set after MT5 import
WARMUP_BARS  = CFG.warmup_bars + 50   # extra buffer so EWM is fully warm
MAGIC        = 202406        # unique identifier for orders placed by this bot

# Direction map: model output → human label
DIR_LABEL = {0: "FLAT", 1: "BUY", -1: "SELL"}


# ══════════════════════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, vecnorm_path: str):
    """Load PPO model + VecNormalize stats (inference-only)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym

    # Minimal dummy env so VecNormalize can be loaded without real data
    n_features = 25   # market features
    n_pos      = 6    # position state features
    obs_dim    = n_features + n_pos

    class _DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
            self.action_space      = gym.spaces.MultiDiscrete([3,
                                                               len(CFG.sl_atr_multipliers),
                                                               len(CFG.tp_r_multipliers)])
        def reset(self, **_):  return np.zeros(obs_dim, dtype=np.float32), {}
        def step(self, _):     return np.zeros(obs_dim, dtype=np.float32), 0.0, True, False, {}

    raw   = DummyVecEnv([_DummyEnv])
    venv  = VecNormalize.load(vecnorm_path, raw)
    venv.training    = False
    venv.norm_reward = False

    # lr_schedule and learning_rate are saved as Python lambdas via cloudpickle.
    # When the model was trained on Python 3.9 and loaded on Python 3.11+,
    # the code object layout changed and cloudpickle cannot deserialize them.
    # We override both with a constant that matches the original linear schedule
    # end-value (0.0) — irrelevant for inference since no gradient steps are taken.
    lr = CFG.ppo_learning_rate  # 6e-5
    custom_objects = {
        "learning_rate": lr,
        "lr_schedule": lambda _: lr,
    }

    model = PPO.load(model_path, env=venv, custom_objects=custom_objects)
    log.info("Model loaded: %s", model_path)
    log.info("VecNorm loaded: %s", vecnorm_path)
    return model, venv


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering  (mirrors features.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "close_ema20_atr", "close_ema50_atr", "close_ema200_atr",
    "ema20_ema50_atr",
    "macd_hist_atr", "roc20_atr",
    "atr_close", "atr_fast_slow", "bb_width_close",
    "range_atr", "upper_wick_ratio", "lower_wick_ratio",
    "ret1_atr", "ret2_atr", "ret3_atr", "ret4_atr", "ret5_atr",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    "session_asia", "session_london", "session_newyork", "session_london_ny_overlap",
]


def bars_to_df(rates) -> pd.DataFrame:
    """Convert MT5 rates array to OHLCV DataFrame with UTC-aware index."""
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"open": "Open", "high": "High",
                             "low": "Low",  "close": "Close",
                             "tick_volume": "Volume"})
    df = df.set_index("time").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Return (feature_df, latest_atr) for the most recent bar."""
    feat, _ = add_stationary_features(df,
                                      atr_period=CFG.atr_period,
                                      rsi_period=CFG.rsi_period,
                                      ema_spans=CFG.ema_spans)
    feat = feat.dropna(subset=FEATURE_COLS + ["atr"])
    if feat.empty:
        raise RuntimeError("Not enough bars to compute features — need more warmup data.")
    return feat, float(feat["atr"].iloc[-1])


# ══════════════════════════════════════════════════════════════════════════════
# Position state (mirrors env_bracket.py _position_state_features)
# ══════════════════════════════════════════════════════════════════════════════

class LivePosition:
    """Tracks the single open bracket position placed by this bot."""

    def __init__(self):
        self.direction:    int   = 0
        self.entry_price:  float = 0.0
        self.sl:           float = 0.0
        self.tp:           float = 0.0
        self.tp_r:         float = 0.0
        self.risk_cash:    float = 0.0
        self.units:        float = 0.0
        self.bars_in_trade: int  = 0
        self.ticket:       Optional[int] = None

    def reset(self):
        self.__init__()

    def position_features(self, close: float, atr: float) -> np.ndarray:
        if self.direction == 0:
            return np.zeros(6, dtype=np.float32)
        eps = 1e-12
        unrealized    = (close - self.entry_price) * self.units * self.direction
        unrealized_r  = unrealized / max(self.risk_cash, eps)
        dist_tp_atr   = ((self.tp - close) * self.direction) / max(atr, eps)
        dist_sl_atr   = ((close - self.sl) * self.direction) / max(atr, eps)
        return np.array([
            float(self.direction),
            unrealized_r,
            min(self.bars_in_trade / 100.0, 10.0),
            dist_tp_atr,
            dist_sl_atr,
            self.tp_r,
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MT5 order helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_current_price(mt5, symbol: str, direction: int) -> float:
    """Return ask (BUY) or bid (SELL) price."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Cannot get tick for {symbol}")
    return tick.ask if direction == 1 else tick.bid


def point_size(mt5, symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Cannot get symbol info for {symbol}")
    return info.point


def normalize_price(price: float, mt5, symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 2
    return round(price, digits)


def calc_lot_size(mt5, symbol: str, risk_cash: float, sl_dist_price: float) -> float:
    """Convert risk_cash / sl_dist to a lot size respecting broker limits."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"No symbol info for {symbol}")

    # For XAUUSD: 1 lot = 100 oz; contract_size gives oz per lot
    contract_size = info.trade_contract_size    # e.g. 100 for XAUUSD
    tick_value    = info.trade_tick_value       # USD per tick
    tick_size     = info.trade_tick_size        # price units per tick

    if sl_dist_price <= 0 or tick_size <= 0:
        return info.volume_min

    value_per_lot = (sl_dist_price / tick_size) * tick_value
    lots = risk_cash / max(value_per_lot, 1e-8)
    lots = max(info.volume_min, min(info.volume_max, round(lots / info.volume_step) * info.volume_step))
    return lots


def close_position(mt5, pos: LivePosition, symbol: str, dry_run: bool) -> bool:
    """Close the tracked open position at market."""
    if pos.direction == 0 or pos.ticket is None:
        return True

    direction_close = -pos.direction   # opposite side to close
    price = get_current_price(mt5, symbol, direction_close)

    order_type = mt5.ORDER_TYPE_SELL if pos.direction == 1 else mt5.ORDER_TYPE_BUY
    request = {
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   symbol,
        "volume":   pos.units,
        "type":     order_type,
        "position": pos.ticket,
        "price":    price,
        "deviation": 20,
        "magic":    MAGIC,
        "comment":  "RL_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if dry_run:
        log.info("[DRY-RUN] Would close position ticket=%s at %.2f", pos.ticket, price)
        return True

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Close failed: %s  retcode=%s", result.comment, result.retcode)
        return False
    deals = mt5.history_deals_get(position=pos.ticket)
    pnl = sum(d.profit for d in deals) if deals else float("nan")
    log.info("Closed ticket=%s deal=%s at %.2f  PnL=%.2f", pos.ticket, result.deal, price, pnl)
    return True


def open_bracket(mt5, pos: LivePosition, symbol: str,
                 direction: int, sl_price: float, tp_price: float,
                 lots: float, dry_run: bool) -> bool:
    """Open a new bracket order with hard SL and TP."""
    price = get_current_price(mt5, symbol, direction)
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

    sl_price = normalize_price(sl_price, mt5, symbol)
    tp_price = normalize_price(tp_price, mt5, symbol)

    request = {
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   symbol,
        "volume":   lots,
        "type":     order_type,
        "price":    price,
        "sl":       sl_price,
        "tp":       tp_price,
        "deviation": 20,
        "magic":    MAGIC,
        "comment":  "RL_bracket",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if dry_run:
        log.info("[DRY-RUN] Would open %s  lots=%.2f  price≈%.2f  SL=%.2f  TP=%.2f",
                 DIR_LABEL[direction], lots, price, sl_price, tp_price)
        return True

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Open failed: %s  retcode=%s", result.comment, result.retcode)
        return False

    pos.ticket = result.order
    log.info("Opened %s ticket=%s  lots=%.2f  entry≈%.2f  SL=%.2f  TP=%.2f",
             DIR_LABEL[direction], pos.ticket, lots, price, sl_price, tp_price)
    return True


def sync_position_from_mt5(mt5, pos: LivePosition, symbol: str, equity: float) -> None:
    """Reconcile local position state with MT5 open positions."""
    positions = mt5.positions_get(symbol=symbol, magic=MAGIC)
    if not positions:
        if pos.direction != 0:
            log.info("Position closed externally (SL/TP hit or manual).")
        pos.reset()
        return

    p = positions[0]   # only one position at a time
    direction = 1 if p.type == mt5.ORDER_TYPE_BUY else -1
    if pos.ticket != p.ticket:
        # Re-sync if ticket changed (e.g. after MT5 restart)
        pos.ticket       = p.ticket
        pos.direction    = direction
        pos.entry_price  = p.price_open
        pos.sl           = p.sl
        pos.tp           = p.tp
        pos.units        = p.volume
        pos.risk_cash    = equity * CFG.risk_fraction
        sl_dist = abs(p.price_open - p.sl)
        pos.tp_r = abs(p.tp - p.price_open) / max(sl_dist, 1e-8)
        log.info("Re-synced position from MT5: ticket=%s dir=%s entry=%.2f",
                 p.ticket, DIR_LABEL[direction], p.price_open)


# ══════════════════════════════════════════════════════════════════════════════
# Main decision loop
# ══════════════════════════════════════════════════════════════════════════════

def make_decision(model, venv, feat_row: pd.Series, atr: float,
                  pos: LivePosition, equity: float) -> tuple[int, int, int]:
    """Run one inference step. Returns (direction, sl_idx, tp_idx)."""
    market_obs = feat_row[FEATURE_COLS].astype(float).to_numpy(dtype=np.float32)
    close      = float(feat_row["Close"])
    pos_obs    = pos.position_features(close, atr)

    obs  = np.concatenate([market_obs, pos_obs]).astype(np.float32)
    obs  = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    obs  = venv.normalize_obs(obs[np.newaxis, :])   # apply VecNorm

    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action[0], dtype=int)
    direction_raw, sl_idx, tp_idx = int(action[0]), int(action[1]), int(action[2])
    desired_direction = {0: 0, 1: 1, 2: -1}[direction_raw]
    return desired_direction, sl_idx, tp_idx


def run_on_bar(mt5, model, venv, pos: LivePosition,
               symbol: str, equity: float, dry_run: bool) -> None:
    """Called once per completed H1 bar."""
    # 1. Fetch enough H1 bars for warmup
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_H1, 0, WARMUP_BARS + 10)
    if rates is None or len(rates) < WARMUP_BARS:
        log.warning("Not enough bars (%s) — skipping.", len(rates) if rates else 0)
        return

    df       = bars_to_df(rates)
    df       = df.iloc[:-1]          # exclude the still-forming current bar
    feat, latest_atr = compute_features(df)
    feat_row = feat.iloc[-1]
    close    = float(feat_row["Close"])
    bar_time = feat.index[-1]

    # 2. Reconcile local state with MT5
    sync_position_from_mt5(mt5, pos, symbol, equity)
    if pos.direction != 0:
        pos.bars_in_trade += 1

    # 3. Get model action
    desired_dir, sl_idx, tp_idx = make_decision(
        model, venv, feat_row, latest_atr, pos, equity)

    sl_mult = CFG.sl_atr_multipliers[sl_idx]
    tp_r    = CFG.tp_r_multipliers[tp_idx]
    sl_dist = sl_mult * latest_atr

    log.info("Bar %s | close=%.2f | atr=%.4f | action=%s sl_mult=%.1f tp_r=%.1f | position=%s",
             bar_time, close, latest_atr, DIR_LABEL[desired_dir],
             sl_mult, tp_r, DIR_LABEL[pos.direction])

    # 4. Execute
    # Close if model wants flat or wants to flip
    if pos.direction != 0 and (desired_dir == 0 or desired_dir != pos.direction):
        if not close_position(mt5, pos, symbol, dry_run):
            return
        pos.reset()

    # Open new position if model wants exposure
    if desired_dir != 0 and pos.direction == 0:
        entry  = close + desired_dir * (CFG.spread_price / 2.0 + CFG.slippage_price)
        sl     = entry - desired_dir * sl_dist
        tp     = entry + desired_dir * tp_r * sl_dist
        risk_cash = equity * CFG.risk_fraction
        lots   = calc_lot_size(mt5, symbol, risk_cash, sl_dist)

        if not dry_run:
            info   = mt5.account_info()
            equity = info.equity if info else equity

        success = open_bracket(mt5, pos, symbol, desired_dir, sl, tp, lots, dry_run)
        if success and not dry_run:
            pos.direction   = desired_dir
            pos.entry_price = entry
            pos.sl          = sl
            pos.tp          = tp
            pos.tp_r        = tp_r
            pos.risk_cash   = risk_cash
            pos.units       = lots
            pos.bars_in_trade = 0


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MT5 PPO XAUUSD H1 Connector")
    p.add_argument("--model",    default="models/best_model/best_model.zip")
    p.add_argument("--vecnorm",  default="models/best_model/best_model_vecnorm.pkl")
    p.add_argument("--login",    type=int, default=None,
                   help="MT5 account number (overrides MT5_LOGIN env var)")
    p.add_argument("--password", default=None,
                   help="MT5 password (overrides MT5_PASSWORD env var)")
    p.add_argument("--server",   default=None,
                   help="MT5 broker server (overrides MT5_SERVER env var)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Compute signals and log them but send no real orders")
    return p.parse_args()


def _resolve_credentials(args) -> dict:
    """Merge env vars and CLI args; CLI takes precedence. Raises if login missing."""
    login    = args.login    or (int(os.environ["MT5_LOGIN"])   if os.environ.get("MT5_LOGIN")    else None)
    password = args.password or os.environ.get("MT5_PASSWORD")
    server   = args.server   or os.environ.get("MT5_SERVER")

    if login is None:
        raise SystemExit(
            "MT5 login not set. Use --login or set the MT5_LOGIN environment variable."
        )
    if password is None:
        raise SystemExit(
            "MT5 password not set. Use --password or set the MT5_PASSWORD environment variable."
        )

    kwargs: dict = {"login": login, "password": password}
    if server:
        kwargs["server"] = server
    return kwargs


def main():
    args = parse_args()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise SystemExit("MetaTrader5 package not found. Run: pip install MetaTrader5")

    global TIMEFRAME_H1
    TIMEFRAME_H1 = mt5.TIMEFRAME_H1

    # Connect to MT5 using env vars or CLI args
    init_kwargs = _resolve_credentials(args)
    log.info("Connecting to MT5 as account #%s on server '%s' …",
             init_kwargs["login"], init_kwargs.get("server", "default"))

    if not mt5.initialize(**init_kwargs):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")

    log.info("Connected to MT5: %s", mt5.terminal_info().name)
    acc = mt5.account_info()
    log.info("Account: #%s  balance=%.2f  equity=%.2f  currency=%s",
             acc.login, acc.balance, acc.equity, acc.currency)

    if args.dry_run:
        log.info("DRY-RUN mode — no real orders will be sent.")

    # Load model
    model, venv = load_model(args.model, args.vecnorm)

    # Position state
    pos = LivePosition()

    log.info("Waiting for H1 bar closes on %s …", SYMBOL)
    last_bar_time: Optional[datetime] = None

    try:
        while True:
            # Check if a new H1 bar has closed
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_H1, 0, 2)
            if rates is None or len(rates) < 2:
                time.sleep(5)
                continue

            # The most recently CLOSED bar is index -2 (index -1 is still forming)
            bar_dt = datetime.fromtimestamp(rates[-2]["time"], tz=timezone.utc)

            if bar_dt != last_bar_time:
                last_bar_time = bar_dt
                acc    = mt5.account_info()
                equity = acc.equity if acc else 10_000.0
                log.info("═══ New H1 bar closed: %s  equity=%.2f ═══", bar_dt, equity)
                try:
                    run_on_bar(mt5, model, venv, pos, SYMBOL, equity, args.dry_run)
                except Exception as exc:
                    log.exception("Error on bar %s: %s", bar_dt, exc)

            time.sleep(10)   # poll every 10 s

    except KeyboardInterrupt:
        log.info("Shutting down …")
    finally:
        mt5.shutdown()
        log.info("MT5 disconnected.")


if __name__ == "__main__":
    main()
