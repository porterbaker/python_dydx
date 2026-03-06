# ============================================================
# BTC PERPS INSTITUTIONAL RESEARCH STACK - REVISED
# ------------------------------------------------------------
# Upgrades in this version:
# - fixed-risk backtesting by default
# - optional probability-weighted sizing
# - threshold sweep utilities
# - top-N per day / week selection
# - transition-only / regime filtering
# - tighter candidate generation
# - safer feature importance / SHAP handling
# - walk-forward event scoring with usable window sizes
# ============================================================

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

HAS_XGB = False
HAS_LGBM = False
HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    LGBMClassifier = None

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None


# ============================================================
# CONFIG
# ============================================================

@dataclass
class DataConfig:
    input_file: str = "./data/historical/BTC-USD_candles_2026-03-06.parquet"


@dataclass
class FeatureConfig:
    atr_len: int = 14
    rv_short: int = 12
    rv_long: int = 48
    zscore_window: int = 96
    oi_window: int = 96
    vol_window: int = 96
    funding_window: int = 96
    ema_fast: int = 20
    ema_slow: int = 50
    adx_len: int = 14
    sweep_atr_mult: float = 1.2
    pivot_cooldown: int = 8

    # tighter candidate generation
    zscore_event_threshold: float = 1.5
    dist_ema_event_threshold: float = 0.004
    min_adx_for_event: float = 18.0
    min_vol_spike_for_transition: float = 1.2


@dataclass
class LabelConfig:
    stop_atr: float = 1.2
    target_atr: float = 2.2
    max_hold: int = 24


@dataclass
class ModelConfig:
    model_family: str = "auto"   # auto | xgb | lgbm | rf
    random_state: int = 42
    prob_threshold: float = 0.62
    top_k_list: tuple[int, ...] = (10, 20, 50, 100)
    min_train_events: int = 300
    min_test_events: int = 50


@dataclass
class CVConfig:
    n_splits: int = 5
    purge_bars: int = 24
    embargo_bars: int = 24


@dataclass
class WalkForwardConfig:
    train_events: int = 4000
    test_events: int = 1000
    step_events: int = 1000
    min_events_per_side: int = 200


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    base_risk_fraction: float = 0.005
    max_risk_fraction: float = 0.02
    min_prob_edge: float = 0.50
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    bars_per_year: int = 288 * 365
    min_entry_gap_bars: int = 3

    # backtest policy
    position_sizing_mode: str = "fixed"   # fixed | prob_weighted
    selection_mode: str = "threshold"     # threshold | top_n_per_day | top_n_per_week
    top_n: int = 2
    regime_filter_mode: str = "all"       # all | transition_only | non_transition | trend_only
    allow_long: bool = False
    allow_short: bool = True


DATA_CFG = DataConfig()
FEAT_CFG = FeatureConfig()
LABEL_CFG = LabelConfig()
MODEL_CFG = ModelConfig()
CV_CFG = CVConfig()
WF_CFG = WalkForwardConfig()
BT_CFG = BacktestConfig()


# ============================================================
# IO
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    if "startedAt" not in df.columns:
        raise ValueError("Expected 'startedAt' column in parquet.")

    df["startedAt"] = pd.to_datetime(df["startedAt"])

    numeric_cols = [
        "open", "high", "low", "close",
        "baseTokenVolume", "usdVolume", "trades",
        "startingOpenInterest", "fundingRate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    df = df.sort_values("startedAt").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


# ============================================================
# BASIC HELPERS
# ============================================================

def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def compute_atr(df: pd.DataFrame, atr_len: int) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr"] = df["tr"].rolling(atr_len).mean()
    return df


def rolling_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean()


# ============================================================
# FEATURE PIPELINE
# ============================================================

def build_quant_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    df = df.copy()
    df = compute_atr(df, cfg.atr_len)

    # returns / momentum
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_2"] = df["close"].pct_change(2)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)

    # trend
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    df["ema_fast_slope"] = df["ema_fast"].diff(3)
    df["ema_slow_slope"] = df["ema_slow"].diff(5)
    df["dist_ema_fast"] = (df["close"] - df["ema_fast"]) / df["ema_fast"]
    df["dist_ema_slow"] = (df["close"] - df["ema_slow"]) / df["ema_slow"]

    close_mean = df["close"].rolling(cfg.zscore_window).mean()
    close_std = df["close"].rolling(cfg.zscore_window).std()
    df["zscore_close"] = (df["close"] - close_mean) / close_std

    df["adx"] = rolling_adx(df, cfg.adx_len)

    # volatility
    df["rv_short"] = df["ret_1"].rolling(cfg.rv_short).std()
    df["rv_long"] = df["ret_1"].rolling(cfg.rv_long).std()
    df["rv_ratio"] = df["rv_short"] / df["rv_long"]
    df["atr_pct"] = df["atr"] / df["close"]
    df["vol_regime_z"] = rolling_z(df["rv_short"], cfg.zscore_window)

    # candle anatomy
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_range"] = candle_range
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_frac"] = df["body"] / candle_range
    df["body_atr"] = df["body"] / df["atr"]
    df["upper_wick_frac"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_frac"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["close_location"] = (df["close"] - df["low"]) / candle_range

    # liquidity / participation
    if "usdVolume" in df.columns:
        df["usdVolume_z"] = rolling_z(df["usdVolume"], cfg.vol_window)
        df["vol_spike"] = df["usdVolume"] / df["usdVolume"].rolling(cfg.vol_window).median()
    else:
        df["usdVolume_z"] = np.nan
        df["vol_spike"] = np.nan

    if "trades" in df.columns:
        df["trades_z"] = rolling_z(df["trades"], cfg.vol_window)
        df["trades_spike"] = df["trades"] / df["trades"].rolling(cfg.vol_window).median()
    else:
        df["trades_z"] = np.nan
        df["trades_spike"] = np.nan

    # OI / positioning
    if "startingOpenInterest" in df.columns:
        df["oi_delta"] = df["startingOpenInterest"].diff()
        df["oi_ret"] = df["startingOpenInterest"].pct_change()
        df["oi_z"] = rolling_z(df["startingOpenInterest"], cfg.oi_window)
        df["oi_delta_z"] = rolling_z(df["oi_delta"], cfg.oi_window)
        df["oi_norm"] = df["startingOpenInterest"] / df["startingOpenInterest"].rolling(cfg.oi_window).median()
        df["price_oi_interact_1"] = df["ret_1"] * df["oi_ret"]
        df["price_oi_interact_3"] = df["ret_3"] * df["oi_ret"]

        df["long_crowded_proxy"] = ((df["ret_3"] > 0) & (df["oi_delta"] > 0)).astype(int)
        df["short_crowded_proxy"] = ((df["ret_3"] < 0) & (df["oi_delta"] > 0)).astype(int)
        df["long_unwind_proxy"] = ((df["ret_3"] < 0) & (df["oi_delta"] < 0)).astype(int)
        df["short_unwind_proxy"] = ((df["ret_3"] > 0) & (df["oi_delta"] < 0)).astype(int)
    else:
        fill_cols = [
            "oi_delta", "oi_ret", "oi_z", "oi_delta_z", "oi_norm",
            "price_oi_interact_1", "price_oi_interact_3",
            "long_crowded_proxy", "short_crowded_proxy",
            "long_unwind_proxy", "short_unwind_proxy",
        ]
        for col in fill_cols:
            df[col] = np.nan

    # funding
    if "fundingRate" in df.columns:
        df["funding_z"] = rolling_z(df["fundingRate"], cfg.funding_window)
        df["funding_abs"] = df["fundingRate"].abs()
        df["funding_oi_interact"] = df["fundingRate"] * df["oi_norm"]
    else:
        df["fundingRate"] = np.nan
        df["funding_z"] = np.nan
        df["funding_abs"] = np.nan
        df["funding_oi_interact"] = np.nan

    # liquidation / impulse proxies
    df["down_impulse"] = (
        (df["ret_1"] < 0) &
        (df["body_atr"] > 1.5) &
        (df["close"] < df["open"])
    ).astype(int)

    df["up_impulse"] = (
        (df["ret_1"] > 0) &
        (df["body_atr"] > 1.5) &
        (df["close"] > df["open"])
    ).astype(int)

    df["flush_proxy"] = (
        (df["ret_1"] < 0) &
        (df["oi_delta"] < 0) &
        (df["vol_spike"] > 1.2)
    ).astype(int)

    df["squeeze_proxy"] = (
        (df["ret_1"] > 0) &
        (df["oi_delta"] < 0) &
        (df["vol_spike"] > 1.2)
    ).astype(int)

    # session / time
    ts = pd.to_datetime(df["startedAt"])
    hour = ts.dt.hour
    dow = ts.dt.dayofweek

    df["hour"] = hour
    df["dayofweek"] = dow
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["session_asia"] = hour.isin([0, 1, 2, 3, 4, 5, 6, 7]).astype(int)
    df["session_europe"] = hour.isin([7, 8, 9, 10, 11, 12, 13]).astype(int)
    df["session_us"] = hour.isin([13, 14, 15, 16, 17, 18, 19, 20]).astype(int)
    df["is_weekend"] = (dow >= 5).astype(int)

    # local range / breakout context
    df["high_break_20"] = (df["high"] > df["high"].shift(1).rolling(20).max()).astype(int)
    df["low_break_20"] = (df["low"] < df["low"].shift(1).rolling(20).min()).astype(int)
    df["range_6"] = (df["high"].rolling(6).max() - df["low"].rolling(6).min()) / df["close"]
    df["range_24"] = (df["high"].rolling(24).max() - df["low"].rolling(24).min()) / df["close"]
    df["range_ratio"] = df["range_6"] / df["range_24"]

    rolling_high_20 = df["high"].shift(1).rolling(20).max()
    rolling_low_20 = df["low"].shift(1).rolling(20).min()
    df["dist_prev20_high"] = (df["close"] - rolling_high_20) / df["close"]
    df["dist_prev20_low"] = (df["close"] - rolling_low_20) / df["close"]

    # defaults
    required_defaults = {
        "zscore_close": np.nan,
        "vol_spike": np.nan,
        "usdVolume_z": np.nan,
        "trades_z": np.nan,
        "trades_spike": np.nan,
        "oi_delta": np.nan,
        "oi_ret": np.nan,
        "oi_z": np.nan,
        "oi_delta_z": np.nan,
        "oi_norm": np.nan,
        "fundingRate": np.nan,
        "funding_z": np.nan,
        "funding_abs": np.nan,
        "funding_oi_interact": np.nan,
        "flush_proxy": 0,
        "squeeze_proxy": 0,
    }
    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    return df


# ============================================================
# REGIME + TIMING LAYER
# ============================================================

def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = "transition"

    uptrend = (
        (df["close"] > df["ema_slow"]) &
        (df["ema_fast"] > df["ema_slow"]) &
        (df["ema_fast_slope"] > 0) &
        (df["ema_slow_slope"] > 0) &
        (df["adx"] > FEAT_CFG.min_adx_for_event)
    )

    downtrend = (
        (df["close"] < df["ema_slow"]) &
        (df["ema_fast"] < df["ema_slow"]) &
        (df["ema_fast_slope"] < 0) &
        (df["ema_slow_slope"] < 0) &
        (df["adx"] > FEAT_CFG.min_adx_for_event)
    )

    df.loc[uptrend, "regime"] = "uptrend"
    df.loc[downtrend, "regime"] = "downtrend"
    df["regime_uptrend"] = (df["regime"] == "uptrend").astype(int)
    df["regime_downtrend"] = (df["regime"] == "downtrend").astype(int)
    df["regime_transition"] = (df["regime"] == "transition").astype(int)
    return df


def detect_secondary_timing_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    df = df.copy()
    df["pivot_high"] = False
    df["pivot_low"] = False
    df["sweep_high"] = False
    df["sweep_low"] = False
    df["sweep_size_atr"] = np.nan
    df["pivot_age_bars"] = np.nan

    if len(df) <= cfg.atr_len:
        return df

    direction = None
    start_i = cfg.atr_len
    extreme_high = df.at[start_i, "high"]
    extreme_high_idx = start_i
    extreme_low = df.at[start_i, "low"]
    extreme_low_idx = start_i
    confirmed_high_pivot = None
    confirmed_low_pivot = None
    last_high_sweep_bar = {}
    last_low_sweep_bar = {}

    for i in range(start_i + 1, len(df)):
        high = df.at[i, "high"]
        low = df.at[i, "low"]
        close = df.at[i, "close"]
        open_ = df.at[i, "open"]
        atr = df.at[i, "atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        threshold = atr * cfg.sweep_atr_mult

        if direction is None:
            if high >= extreme_low + threshold:
                direction = "up"
                extreme_high = high
                extreme_high_idx = i
            elif low <= extreme_high - threshold:
                direction = "down"
                extreme_low = low
                extreme_low_idx = i
            continue

        if direction == "up":
            if high > extreme_high:
                extreme_high = high
                extreme_high_idx = i
            if low <= extreme_high - threshold:
                df.at[extreme_high_idx, "pivot_high"] = True
                confirmed_high_pivot = (extreme_high_idx, extreme_high)
                direction = "down"
                extreme_low = low
                extreme_low_idx = i

        elif direction == "down":
            if low < extreme_low:
                extreme_low = low
                extreme_low_idx = i
            if high >= extreme_low + threshold:
                df.at[extreme_low_idx, "pivot_low"] = True
                confirmed_low_pivot = (extreme_low_idx, extreme_low)
                direction = "up"
                extreme_high = high
                extreme_high_idx = i

        candle_range = max(high - low, 1e-9)
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low

        if confirmed_high_pivot is not None:
            pivot_idx, pivot_price = confirmed_high_pivot
            last_bar = last_high_sweep_bar.get(pivot_idx, -999999)
            if (
                i != pivot_idx and
                i - last_bar >= cfg.pivot_cooldown and
                high > pivot_price and
                close < pivot_price and
                upper_wick / candle_range > 0.20
            ):
                df.at[i, "sweep_high"] = True
                df.at[i, "sweep_size_atr"] = (high - pivot_price) / atr
                df.at[i, "pivot_age_bars"] = i - pivot_idx
                last_high_sweep_bar[pivot_idx] = i

        if confirmed_low_pivot is not None:
            pivot_idx, pivot_price = confirmed_low_pivot
            last_bar = last_low_sweep_bar.get(pivot_idx, -999999)
            if (
                i != pivot_idx and
                i - last_bar >= cfg.pivot_cooldown and
                low < pivot_price and
                close > pivot_price and
                lower_wick / candle_range > 0.20
            ):
                df.at[i, "sweep_low"] = True
                df.at[i, "sweep_size_atr"] = (pivot_price - low) / atr
                df.at[i, "pivot_age_bars"] = i - pivot_idx
                last_low_sweep_bar[pivot_idx] = i

    df["is_sweep_event"] = (df["sweep_high"] | df["sweep_low"]).astype(int)
    return df


# ============================================================
# BROAD EVENT GENERATION - TIGHTENED
# ============================================================

def build_candidate_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["candidate_long"] = False
    df["candidate_short"] = False

    zthr = FEAT_CFG.zscore_event_threshold
    dthr = FEAT_CFG.dist_ema_event_threshold
    adx_thr = FEAT_CFG.min_adx_for_event

    # stricter: regime-aligned only, no transition in base candidates
    long_cond = (
        (df["regime"] == "uptrend") &
        (df["adx"] > adx_thr) &
        (
            (df["flush_proxy"] == 1) |
            (df["sweep_low"]) |
            (df["zscore_close"] < -zthr) |
            (df["dist_ema_fast"] < -dthr)
        )
    )

    short_cond = (
        (df["regime"] == "downtrend") &
        (df["adx"] > adx_thr) &
        (
            (df["squeeze_proxy"] == 1) |
            (df["sweep_high"]) |
            (df["zscore_close"] > zthr) |
            (df["dist_ema_fast"] > dthr)
        )
    )

    df.loc[long_cond.fillna(False), "candidate_long"] = True
    df.loc[short_cond.fillna(False), "candidate_short"] = True

    both = df["candidate_long"] & df["candidate_short"]
    df.loc[both, "candidate_long"] = False
    df.loc[both, "candidate_short"] = False
    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = build_quant_features(df, FEAT_CFG)
    df = classify_regime(df)
    df = detect_secondary_timing_features(df, FEAT_CFG)
    df = build_candidate_events(df)
    return df


# ============================================================
# LABELING
# ============================================================

def triple_barrier_label(
    df: pd.DataFrame,
    event_idx: int,
    side: str,
    stop_atr_mult: float,
    target_atr_mult: float,
    max_holding_bars: int,
) -> tuple[int, float, int]:
    entry_idx = event_idx + 1
    if entry_idx >= len(df):
        return 0, np.nan, event_idx

    entry_price = df.at[entry_idx, "open"]
    atr = df.at[event_idx, "atr"]

    if pd.isna(atr) or atr <= 0:
        return 0, np.nan, event_idx

    if side == "long":
        stop_price = entry_price - stop_atr_mult * atr
        target_price = entry_price + target_atr_mult * atr
        one_r = entry_price - stop_price
    else:
        stop_price = entry_price + stop_atr_mult * atr
        target_price = entry_price - target_atr_mult * atr
        one_r = stop_price - entry_price

    last_idx = min(len(df) - 1, entry_idx + max_holding_bars)

    for j in range(entry_idx, last_idx + 1):
        high = df.at[j, "high"]
        low = df.at[j, "low"]

        if side == "long":
            if low <= stop_price:
                return -1, -1.0, j
            if high >= target_price:
                return 1, target_atr_mult / stop_atr_mult, j
        else:
            if high >= stop_price:
                return -1, -1.0, j
            if low <= target_price:
                return 1, target_atr_mult / stop_atr_mult, j

    exit_price = df.at[last_idx, "close"]
    if side == "long":
        realized_r = (exit_price - entry_price) / one_r
    else:
        realized_r = (entry_price - exit_price) / one_r
    return 0, realized_r, last_idx


def get_feature_columns() -> list[str]:
    return [
        "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24",
        "atr", "atr_pct",
        "ema_fast_slope", "ema_slow_slope",
        "dist_ema_fast", "dist_ema_slow",
        "rv_short", "rv_long", "rv_ratio", "vol_regime_z",
        "body_frac", "upper_wick_frac", "lower_wick_frac", "close_location", "body_atr",
        "usdVolume_z", "vol_spike", "trades_z", "trades_spike",
        "oi_delta", "oi_ret", "oi_z", "oi_delta_z", "oi_norm",
        "price_oi_interact_1", "price_oi_interact_3",
        "long_crowded_proxy", "short_crowded_proxy",
        "long_unwind_proxy", "short_unwind_proxy",
        "fundingRate", "funding_z", "funding_abs", "funding_oi_interact",
        "down_impulse", "up_impulse", "flush_proxy", "squeeze_proxy",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "session_asia", "session_europe", "session_us", "is_weekend",
        "sweep_high", "sweep_low", "sweep_size_atr", "pivot_age_bars", "is_sweep_event",
        "range_6", "range_24", "range_ratio",
        "dist_prev20_high", "dist_prev20_low",
        "high_break_20", "low_break_20",
        "regime_uptrend", "regime_downtrend", "regime_transition",
    ]


def build_event_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = list(dict.fromkeys(get_feature_columns()))

    long_events = df.index[df["candidate_long"]].tolist()
    short_events = df.index[df["candidate_short"]].tolist()

    for event_idx, side in [(i, "long") for i in long_events] + [(i, "short") for i in short_events]:
        label, realized_r, exit_idx = triple_barrier_label(
            df=df,
            event_idx=event_idx,
            side=side,
            stop_atr_mult=LABEL_CFG.stop_atr,
            target_atr_mult=LABEL_CFG.target_atr,
            max_holding_bars=LABEL_CFG.max_hold,
        )

        row = {
            "event_idx": event_idx,
            "startedAt": df.at[event_idx, "startedAt"],
            "side": side,
            "target_hit": int(label == 1),
            "tb_label": label,
            "realized_r": realized_r,
            "exit_idx": exit_idx,
            "entry_regime": df.at[event_idx, "regime"],
            "hour": df.at[event_idx, "hour"],
        }
        for col in feature_cols:
            row[col] = df.at[event_idx, col] if col in df.columns else np.nan

        rows.append(row)

    events = pd.DataFrame(rows).sort_values("startedAt").reset_index(drop=True)
    return events


# ============================================================
# PURGED CV
# ============================================================

def purged_time_series_splits(
    n_samples: int,
    n_splits: int,
    purge_bars: int,
    embargo_bars: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    indices = np.arange(n_samples)
    current = 0

    for fold_size in fold_sizes:
        test_start = current
        test_end = current + fold_size

        test_idx = indices[test_start:test_end]
        left_end = max(0, test_start - purge_bars)
        right_start = min(n_samples, test_end + embargo_bars)

        train_left = indices[:left_end]
        train_right = indices[right_start:]
        train_idx = np.concatenate([train_left, train_right])

        yield train_idx, test_idx
        current = test_end


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(model_family: str, random_state: int):
    family = model_family.lower()

    if family == "auto":
        if HAS_XGB:
            family = "xgb"
        elif HAS_LGBM:
            family = "lgbm"
        else:
            family = "rf"

    if family == "xgb" and HAS_XGB:
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_weight=8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=4,
        )

    if family == "lgbm" and HAS_LGBM:
        return LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_samples=20,
            objective="binary",
            random_state=random_state,
            n_jobs=4,
            verbosity=-1,
        )

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=12,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=4,
    )


# ============================================================
# CV + TRAINING
# ============================================================

def cross_validate_side_model(events_side: pd.DataFrame, model_family: str) -> pd.DataFrame:
    feature_cols = list(dict.fromkeys(get_feature_columns()))
    events_side = events_side.sort_values("startedAt").reset_index(drop=True)

    X = events_side[feature_cols]
    y = events_side["target_hit"].astype(int)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        purged_time_series_splits(
            n_samples=len(events_side),
            n_splits=CV_CFG.n_splits,
            purge_bars=CV_CFG.purge_bars,
            embargo_bars=CV_CFG.embargo_bars,
        ),
        start=1,
    ):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        model = build_model(model_family, MODEL_CFG.random_state + fold)
        model.fit(X_imp[train_idx], y.iloc[train_idx])

        prob = model.predict_proba(X_imp[test_idx])[:, 1]
        y_test = y.iloc[test_idx]

        auc = roc_auc_score(y_test, prob) if y_test.nunique() > 1 else np.nan
        ll = log_loss(y_test, prob, labels=[0, 1])

        rows.append(
            {
                "fold": fold,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "auc": auc,
                "logloss": ll,
            }
        )

    return pd.DataFrame(rows)


def fit_side_model(events_side: pd.DataFrame, side_name: str) -> dict[str, Any]:
    if len(events_side) < MODEL_CFG.min_train_events:
        raise ValueError(f"Not enough {side_name} events to train: {len(events_side)}")

    events_side = events_side.sort_values("startedAt").reset_index(drop=True)
    feature_cols = list(dict.fromkeys(get_feature_columns()))
    feature_cols = [c for c in feature_cols if events_side[c].nunique(dropna=False) > 1]

    missing_cols = [c for c in feature_cols if c not in events_side.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns for {side_name}: {missing_cols}")

    n = len(events_side)
    train_end = int(n * 0.60)
    valid_end = int(n * 0.80)

    train = events_side.iloc[:train_end].copy()
    valid = events_side.iloc[train_end:valid_end].copy()
    test = events_side.iloc[valid_end:].copy()

    X_train = train[feature_cols]
    y_train = train["target_hit"].astype(int)

    X_valid = valid[feature_cols]
    y_valid = valid["target_hit"].astype(int)

    X_test = test[feature_cols]
    y_test = test["target_hit"].astype(int)

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_valid_imp = imp.transform(X_valid)
    X_test_imp = imp.transform(X_test)

    model = build_model(MODEL_CFG.model_family, MODEL_CFG.random_state)
    model.fit(X_train_imp, y_train)

    valid_prob = model.predict_proba(X_valid_imp)[:, 1] if len(valid) > 0 else np.array([])
    test_prob = model.predict_proba(X_test_imp)[:, 1] if len(test) > 0 else np.array([])

    valid_auc = roc_auc_score(y_valid, valid_prob) if len(valid) > 0 and y_valid.nunique() > 1 else np.nan
    test_auc = roc_auc_score(y_test, test_prob) if len(test) > 0 and y_test.nunique() > 1 else np.nan

    best_threshold = MODEL_CFG.prob_threshold
    best_metric = -np.inf
    thresholds = np.arange(0.50, 0.91, 0.01)

    if len(valid) > 0:
        valid_tmp = valid.copy()
        valid_tmp["pred_prob"] = valid_prob

        for th in thresholds:
            picked = valid_tmp[valid_tmp["pred_prob"] >= th]
            if len(picked) < 15:
                continue
            metric = picked["realized_r"].mean() * np.sqrt(len(picked))
            if metric > best_metric:
                best_metric = metric
                best_threshold = float(th)

    # constant cols info
    constant_cols = [c for c in feature_cols if events_side[c].nunique(dropna=False) <= 1]

    # robust feature importance
    feat_imp = pd.DataFrame(columns=["feature", "importance"])
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_).ravel()
        if len(importances) != len(feature_cols):
            print(
                f"\n[WARN] {side_name} feature importance length mismatch: "
                f"{len(importances)} importances vs {len(feature_cols)} feature columns"
            )
            min_len = min(len(importances), len(feature_cols))
            feat_imp = pd.DataFrame(
                {
                    "feature": feature_cols[:min_len],
                    "importance": importances[:min_len],
                }
            ).sort_values("importance", ascending=False)
        else:
            feat_imp = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)

    shap_df = None
    if HAS_SHAP and len(test) > 0:
        try:
            sample_n = min(500, len(test))
            X_sample = X_test.iloc[:sample_n].copy()
            X_sample_imp = imp.transform(X_sample)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_imp)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            mean_abs = np.abs(shap_values).mean(axis=0)
            min_len = min(len(mean_abs), len(feature_cols))
            shap_df = pd.DataFrame(
                {
                    "feature": feature_cols[:min_len],
                    "mean_abs_shap": mean_abs[:min_len],
                }
            ).sort_values("mean_abs_shap", ascending=False)
        except Exception as e:
            print(f"\n[WARN] SHAP failed for {side_name}: {e}")
            shap_df = None

    cv_df = cross_validate_side_model(events_side, MODEL_CFG.model_family)

    return {
        "side": side_name,
        "model": model,
        "imputer": imp,
        "feature_cols": feature_cols,
        "train_events": train,
        "valid_events": valid,
        "test_events": test,
        "valid_auc": valid_auc,
        "test_auc": test_auc,
        "threshold": best_threshold,
        "feature_importance": feat_imp,
        "shap_importance": shap_df,
        "cv_results": cv_df,
        "constant_cols": constant_cols,
    }


def print_model_summary(model_info: dict[str, Any]) -> None:
    print(f"\n=== {model_info['side'].upper()} MODEL ===")
    print(f"Validation AUC: {model_info['valid_auc']}")
    print(f"Test AUC:       {model_info['test_auc']}")
    print(f"Chosen threshold: {model_info['threshold']:.2f}")

    if model_info["constant_cols"]:
        print("\nConstant / degenerate columns:")
        print(model_info["constant_cols"])

    if model_info["cv_results"] is not None and not model_info["cv_results"].empty:
        cv = model_info["cv_results"]
        print("\nPurged CV summary:")
        print(cv.to_string(index=False))
        print(f"Mean CV AUC: {cv['auc'].mean():.4f}")
        print(f"Mean CV LogLoss: {cv['logloss'].mean():.4f}")

    print("\nTop feature importance:")
    print(model_info["feature_importance"].head(15).to_string(index=False))

    if model_info["shap_importance"] is not None:
        print("\nTop SHAP importance:")
        print(model_info["shap_importance"].head(15).to_string(index=False))


# ============================================================
# PROBABILITY DIAGNOSTICS
# ============================================================

def attach_pred_probs(model_info: dict[str, Any]) -> pd.DataFrame:
    test_ev = model_info["test_events"].copy()
    if test_ev.empty:
        return test_ev

    X = test_ev[model_info["feature_cols"]]
    X_imp = model_info["imputer"].transform(X)
    test_ev["pred_prob"] = model_info["model"].predict_proba(X_imp)[:, 1]
    return test_ev


def probability_decile_table(events_df: pd.DataFrame, prob_col: str = "pred_prob") -> pd.DataFrame:
    tmp = events_df.dropna(subset=[prob_col]).copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["rank_tmp"] = tmp[prob_col].rank(method="first")
    tmp["prob_decile"] = pd.qcut(tmp["rank_tmp"], 10, labels=False, duplicates="drop") + 1

    out = (
        tmp.groupby("prob_decile")
        .agg(
            n_events=("target_hit", "size"),
            avg_prob=(prob_col, "mean"),
            hit_rate=("target_hit", "mean"),
            avg_realized_r=("realized_r", "mean"),
            median_realized_r=("realized_r", "median"),
        )
        .reset_index()
        .sort_values("prob_decile")
    )
    out["hit_rate_pct"] = out["hit_rate"] * 100.0
    return out


def precision_at_top_k(events_df: pd.DataFrame, k_list: tuple[int, ...], prob_col: str = "pred_prob") -> pd.DataFrame:
    tmp = events_df.dropna(subset=[prob_col]).sort_values(prob_col, ascending=False).copy()
    rows = []

    if tmp.empty:
        return pd.DataFrame()

    for k in k_list:
        top = tmp.head(k)
        if top.empty:
            continue

        rows.append(
            {
                "top_k": k,
                "n_events": len(top),
                "avg_prob": top[prob_col].mean(),
                "precision": top["target_hit"].mean(),
                "precision_pct": top["target_hit"].mean() * 100.0,
                "avg_realized_r": top["realized_r"].mean(),
                "sum_realized_r": top["realized_r"].sum(),
            }
        )

    return pd.DataFrame(rows)


def cumulative_return_by_ranked_probability(events_df: pd.DataFrame, prob_col: str = "pred_prob") -> pd.DataFrame:
    tmp = events_df.dropna(subset=[prob_col]).sort_values(prob_col, ascending=False).copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["rank"] = np.arange(1, len(tmp) + 1)
    tmp["cum_realized_r"] = tmp["realized_r"].cumsum()
    tmp["cum_hit_rate"] = tmp["target_hit"].expanding().mean()
    return tmp[["rank", prob_col, "realized_r", "cum_realized_r", "cum_hit_rate", "target_hit"]]


def print_prob_diagnostics(title: str, scored_test: pd.DataFrame) -> None:
    print(f"\n=== {title}: PROBABILITY DECILES ===")
    dec = probability_decile_table(scored_test)
    print(dec.to_string(index=False) if not dec.empty else "No decile data.")

    print(f"\n=== {title}: PRECISION AT TOP-K ===")
    pak = precision_at_top_k(scored_test, MODEL_CFG.top_k_list)
    print(pak.to_string(index=False) if not pak.empty else "No top-k data.")

    print(f"\n=== {title}: CUMULATIVE RETURN BY RANKED PROBABILITY (HEAD) ===")
    cum = cumulative_return_by_ranked_probability(scored_test)
    print(cum.head(20).to_string(index=False) if not cum.empty else "No cumulative data.")


# ============================================================
# WALK-FORWARD TRAINING
# ============================================================

def score_side_events(model_info: dict[str, Any], events_side: pd.DataFrame) -> pd.DataFrame:
    scored = events_side.copy()
    X = scored[model_info["feature_cols"]]
    X_imp = model_info["imputer"].transform(X)
    scored["pred_prob"] = model_info["model"].predict_proba(X_imp)[:, 1]
    return scored


def train_side_on_window(events_side_train: pd.DataFrame) -> tuple[Any, Any, list[str]]:
    feature_cols = list(dict.fromkeys(get_feature_columns()))
    X = events_side_train[feature_cols]
    y = events_side_train["target_hit"].astype(int)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    model = build_model(MODEL_CFG.model_family, MODEL_CFG.random_state)
    model.fit(X_imp, y)
    return imp, model, feature_cols


def walk_forward_score_events(events_side: pd.DataFrame) -> pd.DataFrame:
    events_side = events_side.sort_values("startedAt").reset_index(drop=True)
    scored_parts = []

    n = len(events_side)
    train_n = WF_CFG.train_events
    test_n = WF_CFG.test_events
    step_n = WF_CFG.step_events

    start = 0
    while True:
        train_start = start
        train_end = train_start + train_n
        test_end = train_end + test_n

        if test_end > n:
            break

        train_df = events_side.iloc[train_start:train_end].copy()
        test_df = events_side.iloc[train_end:test_end].copy()

        if len(train_df) < WF_CFG.min_events_per_side or len(test_df) < MODEL_CFG.min_test_events:
            start += step_n
            continue

        imp, model, feature_cols = train_side_on_window(train_df)
        X_test = test_df[feature_cols]
        X_test_imp = imp.transform(X_test)
        test_df["pred_prob"] = model.predict_proba(X_test_imp)[:, 1]
        scored_parts.append(test_df)

        start += step_n

    if not scored_parts:
        return pd.DataFrame(columns=events_side.columns.tolist() + ["pred_prob"])

    out = pd.concat(scored_parts, ignore_index=True)
    return out.sort_values("startedAt").reset_index(drop=True)


# ============================================================
# BACKTEST
# ============================================================

def apply_costs(price: float, side: str, action: str, slippage_bps: float, fee_bps: float) -> float:
    total_bps = (slippage_bps + fee_bps) / 10_000.0

    if side == "long":
        if action == "entry":
            return price * (1 + total_bps)
        return price * (1 - total_bps)
    else:
        if action == "entry":
            return price * (1 - total_bps)
        return price * (1 + total_bps)


def probability_weighted_risk(prob: float, cfg: BacktestConfig) -> float:
    edge = max(0.0, prob - cfg.min_prob_edge)
    if edge <= 0:
        return 0.0
    scaled = cfg.base_risk_fraction * (1.0 + 8.0 * edge)
    return min(cfg.max_risk_fraction, scaled)


def choose_risk_fraction(prob: float, cfg: BacktestConfig) -> float:
    if cfg.position_sizing_mode == "prob_weighted":
        return probability_weighted_risk(prob, cfg)
    return cfg.base_risk_fraction


def filter_scored_events_for_backtest(scored_events: pd.DataFrame, cfg: BacktestConfig, prob_threshold: float) -> pd.DataFrame:
    ev = scored_events.copy()

    if ev.empty:
        return ev

    # threshold selection
    ev = ev[ev["pred_prob"] >= prob_threshold].copy()

    # side selection
    if not cfg.allow_long:
        ev = ev[ev["side"] != "long"]
    if not cfg.allow_short:
        ev = ev[ev["side"] != "short"]

    # regime filter
    if cfg.regime_filter_mode == "transition_only":
        ev = ev[ev["entry_regime"] == "transition"]
    elif cfg.regime_filter_mode == "non_transition":
        ev = ev[ev["entry_regime"] != "transition"]
    elif cfg.regime_filter_mode == "trend_only":
        ev = ev[ev["entry_regime"].isin(["uptrend", "downtrend"])]

    ev = ev[
        ((ev["side"] == "short") & (ev["entry_regime"] == "downtrend")) |
        ((ev["side"] == "long") & (ev["entry_regime"] == "uptrend"))
    ]

    if ev.empty:
        return ev

    # selection mode
    if cfg.selection_mode == "top_n_per_day":
        ev["bucket"] = pd.to_datetime(ev["startedAt"]).dt.floor("D")
        ev = (
            ev.sort_values(["bucket", "pred_prob"], ascending=[True, False])
              .groupby("bucket", group_keys=False)
              .head(cfg.top_n)
              .drop(columns=["bucket"])
        )
    elif cfg.selection_mode == "top_n_per_week":
        ev["bucket"] = pd.to_datetime(ev["startedAt"]).dt.to_period("W").astype(str)
        ev = (
            ev.sort_values(["bucket", "pred_prob"], ascending=[True, False])
              .groupby("bucket", group_keys=False)
              .head(cfg.top_n)
              .drop(columns=["bucket"])
        )

    return ev.sort_values("startedAt").reset_index(drop=True)


def backtest_scored_events(
    df: pd.DataFrame,
    scored_events: pd.DataFrame,
    prob_threshold: float,
    bt_cfg: BacktestConfig,
    label_cfg: LabelConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    filtered = filter_scored_events_for_backtest(scored_events, bt_cfg, prob_threshold)

    if filtered.empty:
        empty_eq = pd.DataFrame({"startedAt": df["startedAt"], "equity": bt_cfg.initial_capital})
        return {
            "initial_capital": bt_cfg.initial_capital,
            "final_capital": bt_cfg.initial_capital,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_pnl_per_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_per_trade": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
        }, pd.DataFrame(), empty_eq

    capital = bt_cfg.initial_capital
    trades = []
    equity_rows = []
    last_entry_idx = -10**9

    for _, ev in filtered.iterrows():
        event_idx = int(ev["event_idx"])
        entry_idx = event_idx + 1
        if entry_idx >= len(df):
            continue

        if entry_idx - last_entry_idx < bt_cfg.min_entry_gap_bars:
            continue

        side = ev["side"]
        prob = float(ev["pred_prob"])
        risk_fraction = choose_risk_fraction(prob, bt_cfg)
        if risk_fraction <= 0:
            continue

        entry_price_raw = df.at[entry_idx, "open"]
        entry_price = apply_costs(entry_price_raw, side, "entry", bt_cfg.slippage_bps, bt_cfg.fee_bps)

        atr = df.at[event_idx, "atr"]
        if pd.isna(atr) or atr <= 0:
            continue

        if side == "long":
            stop_price_raw = entry_price_raw - label_cfg.stop_atr * atr
            target_price_raw = entry_price_raw + label_cfg.target_atr * atr
            stop_price = apply_costs(stop_price_raw, side, "exit", bt_cfg.slippage_bps, bt_cfg.fee_bps)
            target_price = apply_costs(target_price_raw, side, "exit", bt_cfg.slippage_bps, bt_cfg.fee_bps)
            risk_per_unit = entry_price - stop_price
        else:
            stop_price_raw = entry_price_raw + label_cfg.stop_atr * atr
            target_price_raw = entry_price_raw - label_cfg.target_atr * atr
            stop_price = apply_costs(stop_price_raw, side, "exit", bt_cfg.slippage_bps, bt_cfg.fee_bps)
            target_price = apply_costs(target_price_raw, side, "exit", bt_cfg.slippage_bps, bt_cfg.fee_bps)
            risk_per_unit = stop_price - entry_price

        if risk_per_unit <= 0:
            continue

        dollars_at_risk = capital * risk_fraction
        qty = dollars_at_risk / risk_per_unit

        exit_idx = min(entry_idx + label_cfg.max_hold, len(df) - 1)
        exit_reason = "time_stop"
        exit_price = None

        for j in range(entry_idx, exit_idx + 1):
            high = df.at[j, "high"]
            low = df.at[j, "low"]

            if side == "long":
                stop_hit = low <= stop_price_raw
                target_hit = high >= target_price_raw
                if stop_hit and target_hit:
                    exit_reason = "stop_and_target_same_bar"
                    exit_price = stop_price
                    exit_idx = j
                    break
                if stop_hit:
                    exit_reason = "stop"
                    exit_price = stop_price
                    exit_idx = j
                    break
                if target_hit:
                    exit_reason = "target"
                    exit_price = target_price
                    exit_idx = j
                    break
            else:
                stop_hit = high >= stop_price_raw
                target_hit = low <= target_price_raw
                if stop_hit and target_hit:
                    exit_reason = "stop_and_target_same_bar"
                    exit_price = stop_price
                    exit_idx = j
                    break
                if stop_hit:
                    exit_reason = "stop"
                    exit_price = stop_price
                    exit_idx = j
                    break
                if target_hit:
                    exit_reason = "target"
                    exit_price = target_price
                    exit_idx = j
                    break

        if exit_price is None:
            raw_close = df.at[exit_idx, "close"]
            exit_price = apply_costs(raw_close, side, "exit", bt_cfg.slippage_bps, bt_cfg.fee_bps)

        if side == "long":
            pnl = (exit_price - entry_price) * qty
            ret_pct = (exit_price / entry_price - 1.0) * 100.0
        else:
            pnl = (entry_price - exit_price) * qty
            ret_pct = (entry_price / exit_price - 1.0) * 100.0

        capital += pnl
        last_entry_idx = entry_idx

        trades.append(
            {
                "side": side,
                "event_time": df.at[event_idx, "startedAt"],
                "entry_time": df.at[entry_idx, "startedAt"],
                "exit_time": df.at[exit_idx, "startedAt"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "prob": prob,
                "risk_fraction": risk_fraction,
                "bars_held": exit_idx - entry_idx,
                "pnl": pnl,
                "return_pct": ret_pct,
                "exit_reason": exit_reason,
                "entry_regime": ev["entry_regime"],
            }
        )

        equity_rows.append({"startedAt": df.at[exit_idx, "startedAt"], "equity": capital})

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        equity_df = pd.DataFrame({"startedAt": df["startedAt"], "equity": bt_cfg.initial_capital})
    else:
        equity_df = pd.DataFrame(equity_rows).sort_values("startedAt").reset_index(drop=True)
        equity_df = pd.concat(
            [
                pd.DataFrame({"startedAt": [df["startedAt"].iloc[0]], "equity": [bt_cfg.initial_capital]}),
                equity_df,
            ],
            ignore_index=True,
        )

    results = summarize_backtest(trades_df, equity_df, bt_cfg)
    return results, trades_df, equity_df


def summarize_backtest(trades_df: pd.DataFrame, equity_df: pd.DataFrame, bt_cfg: BacktestConfig) -> dict[str, Any]:
    if equity_df.empty:
        return {
            "initial_capital": bt_cfg.initial_capital,
            "final_capital": bt_cfg.initial_capital,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_pnl_per_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_per_trade": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
        }

    final_capital = equity_df["equity"].iloc[-1]
    total_return_pct = (final_capital / bt_cfg.initial_capital - 1.0) * 100.0

    eq = equity_df.copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    eq["peak"] = eq["equity"].cummax()
    eq["drawdown"] = eq["equity"] / eq["peak"] - 1.0
    max_drawdown_pct = eq["drawdown"].min() * 100.0

    ret_std = eq["ret"].std()
    sharpe_ratio = 0.0
    if pd.notna(ret_std) and ret_std > 0:
        sharpe_ratio = eq["ret"].mean() / ret_std * math.sqrt(bt_cfg.bars_per_year)

    if trades_df.empty:
        return {
            "initial_capital": bt_cfg.initial_capital,
            "final_capital": final_capital,
            "net_profit": final_capital - bt_cfg.initial_capital,
            "total_return_pct": total_return_pct,
            "profit_factor": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_pnl_per_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_per_trade": 0.0,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown_pct,
        }

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.inf

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
    win_rate_pct = (len(wins) / len(trades_df)) * 100.0
    avg_pnl_per_trade = trades_df["pnl"].mean()
    expectancy_per_trade = avg_pnl_per_trade

    return {
        "initial_capital": bt_cfg.initial_capital,
        "final_capital": final_capital,
        "net_profit": final_capital - bt_cfg.initial_capital,
        "total_return_pct": total_return_pct,
        "profit_factor": profit_factor,
        "total_trades": len(trades_df),
        "win_rate_pct": win_rate_pct,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy_per_trade": expectancy_per_trade,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
    }


def print_backtest_results(results: dict[str, Any], title: str) -> None:
    print(f"\n=== {title} ===")
    for k, v in results.items():
        if isinstance(v, (float, np.floating)):
            print(f"{k:24s}: {v:,.4f}")
        else:
            print(f"{k:24s}: {v}")


# ============================================================
# BACKTEST UTILITIES
# ============================================================

def threshold_sweep(
    df: pd.DataFrame,
    scored_events: pd.DataFrame,
    thresholds: list[float],
    bt_cfg: BacktestConfig,
    label_cfg: LabelConfig,
) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        res, trades, _ = backtest_scored_events(df, scored_events, th, bt_cfg, label_cfg)
        rows.append({
            "threshold": th,
            "total_trades": res["total_trades"],
            "final_capital": res["final_capital"],
            "net_profit": res["net_profit"],
            "total_return_pct": res["total_return_pct"],
            "profit_factor": res["profit_factor"],
            "win_rate_pct": res["win_rate_pct"],
            "avg_pnl_per_trade": res["avg_pnl_per_trade"],
            "max_drawdown_pct": res["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_cumulative_ranked_probability(cum_df: pd.DataFrame, title: str) -> None:
    if cum_df.empty:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(cum_df["rank"], cum_df["cum_realized_r"])
    plt.title(title)
    plt.xlabel("Rank (highest probability first)")
    plt.ylabel("Cumulative realized R")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_equity_curve(equity_df: pd.DataFrame, title: str) -> None:
    if equity_df.empty:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(equity_df["startedAt"], equity_df["equity"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== CONFIG ===")
    print("Data:", asdict(DATA_CFG))
    print("Features:", asdict(FEAT_CFG))
    print("Labels:", asdict(LABEL_CFG))
    print("Model:", asdict(MODEL_CFG))
    print("CV:", asdict(CV_CFG))
    print("WalkForward:", asdict(WF_CFG))
    print("Backtest:", asdict(BT_CFG))

    # 1) pipeline
    df = load_data(DATA_CFG.input_file)
    df = run_pipeline(df)

    print("\n=== PIPELINE COUNTS ===")
    print("candidate_long:", int(df["candidate_long"].sum()))
    print("candidate_short:", int(df["candidate_short"].sum()))
    print("sweep_high:", int(df["sweep_high"].sum()))
    print("sweep_low:", int(df["sweep_low"].sum()))

    # 2) events
    events = build_event_dataset(df)
    long_events = events[events["side"] == "long"].copy()
    short_events = events[events["side"] == "short"].copy()

    print("\n=== EVENT DATASET ===")
    print("Total events :", len(events))
    print("Long events  :", len(long_events))
    print("Short events :", len(short_events))
    if len(events) > 0:
        print("Sweep event share:", f"{events['is_sweep_event'].mean() * 100:.2f}%")

    # 3) side models
    long_model_info = None
    short_model_info = None

    if len(long_events) >= MODEL_CFG.min_train_events:
        long_model_info = fit_side_model(long_events, "long")
        print_model_summary(long_model_info)

    if len(short_events) >= MODEL_CFG.min_train_events:
        short_model_info = fit_side_model(short_events, "short")
        print_model_summary(short_model_info)

    # 4) holdout diagnostics
    long_test_scored = attach_pred_probs(long_model_info) if long_model_info is not None else pd.DataFrame()
    short_test_scored = attach_pred_probs(short_model_info) if short_model_info is not None else pd.DataFrame()

    if not long_test_scored.empty:
        print("\n=== LONG TEST SET: PROBABILITY DECILES ===")
        print(probability_decile_table(long_test_scored).to_string(index=False))
        print("\n=== LONG TEST SET: PRECISION AT TOP-K ===")
        print(precision_at_top_k(long_test_scored, MODEL_CFG.top_k_list).to_string(index=False))
        print("\n=== LONG TEST SET: CUMULATIVE RETURN BY RANKED PROBABILITY (HEAD) ===")
        print(cumulative_return_by_ranked_probability(long_test_scored).head(20).to_string(index=False))
        plot_cumulative_ranked_probability(
            cumulative_return_by_ranked_probability(long_test_scored),
            "Long Test Set: Cumulative R by Ranked Probability",
        )

    if not short_test_scored.empty:
        print("\n=== SHORT TEST SET: PROBABILITY DECILES ===")
        print(probability_decile_table(short_test_scored).to_string(index=False))
        print("\n=== SHORT TEST SET: PRECISION AT TOP-K ===")
        print(precision_at_top_k(short_test_scored, MODEL_CFG.top_k_list).to_string(index=False))
        print("\n=== SHORT TEST SET: CUMULATIVE RETURN BY RANKED PROBABILITY (HEAD) ===")
        print(cumulative_return_by_ranked_probability(short_test_scored).head(20).to_string(index=False))
        plot_cumulative_ranked_probability(
            cumulative_return_by_ranked_probability(short_test_scored),
            "Short Test Set: Cumulative R by Ranked Probability",
        )

    # 5) walk-forward scoring
    wf_long = walk_forward_score_events(long_events) if len(long_events) >= WF_CFG.min_events_per_side else pd.DataFrame()
    wf_short = walk_forward_score_events(short_events) if len(short_events) >= WF_CFG.min_events_per_side else pd.DataFrame()
    wf_scored = pd.concat([wf_long, wf_short], ignore_index=True).sort_values("startedAt").reset_index(drop=True)

    print("\n=== WALK-FORWARD SCORED EVENTS ===")
    print("Total WF scored events:", len(wf_scored))
    if not wf_scored.empty:
        print("\nWF probability deciles:")
        print(probability_decile_table(wf_scored).to_string(index=False))
        print("\nWF precision at top-k:")
        print(precision_at_top_k(wf_scored, MODEL_CFG.top_k_list).to_string(index=False))

    # 6) primary backtest using current config
    wf_results, wf_trades, wf_equity = backtest_scored_events(
        df=df,
        scored_events=wf_scored,
        prob_threshold=MODEL_CFG.prob_threshold,
        bt_cfg=BT_CFG,
        label_cfg=LABEL_CFG,
    )
    print_backtest_results(wf_results, "WALK-FORWARD ML BACKTEST")

    if not wf_trades.empty:
        print("\n=== RECENT TRADES ===")
        print(wf_trades.tail(10).to_string(index=False))

        print("\n=== TRADE BREAKDOWN BY SIDE ===")
        print(
            wf_trades.groupby("side")
            .agg(
                trades=("pnl", "size"),
                net_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("pnl", lambda s: (s > 0).mean() * 100.0),
            )
            .reset_index()
            .to_string(index=False)
        )

        print("\n=== TRADE BREAKDOWN BY REGIME ===")
        print(
            wf_trades.groupby("entry_regime")
            .agg(
                trades=("pnl", "size"),
                net_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("pnl", lambda s: (s > 0).mean() * 100.0),
            )
            .reset_index()
            .to_string(index=False)
        )

    plot_equity_curve(wf_equity, "Walk-Forward ML Equity Curve")

    # 7) threshold sweep - fixed risk, all regimes
    print("\n=== THRESHOLD SWEEP: FIXED RISK / ALL REGIMES ===")
    base_sweep_cfg = BacktestConfig(**asdict(BT_CFG))
    base_sweep_cfg.position_sizing_mode = "fixed"
    base_sweep_cfg.regime_filter_mode = "all"
    base_sweep_cfg.selection_mode = "threshold"

    sweep_df = threshold_sweep(
        df=df,
        scored_events=wf_scored,
        thresholds=[0.56, 0.58, 0.60, 0.62, 0.64, 0.66],
        bt_cfg=base_sweep_cfg,
        label_cfg=LABEL_CFG,
    )
    print(sweep_df.to_string(index=False))

    # 8) threshold sweep - fixed risk, transition only
    print("\n=== THRESHOLD SWEEP: FIXED RISK / TRANSITION ONLY ===")
    trans_cfg = BacktestConfig(**asdict(BT_CFG))
    trans_cfg.position_sizing_mode = "fixed"
    trans_cfg.regime_filter_mode = "transition_only"
    trans_cfg.selection_mode = "threshold"

    trans_sweep_df = threshold_sweep(
        df=df,
        scored_events=wf_scored,
        thresholds=[0.56, 0.58, 0.60, 0.62, 0.64, 0.66],
        bt_cfg=trans_cfg,
        label_cfg=LABEL_CFG,
    )
    print(trans_sweep_df.to_string(index=False))

    # 9) top-N per day - fixed risk
    print("\n=== TOP-N PER DAY BACKTEST (FIXED RISK, ALL REGIMES) ===")
    topn_cfg = BacktestConfig(**asdict(BT_CFG))
    topn_cfg.position_sizing_mode = "fixed"
    topn_cfg.selection_mode = "top_n_per_day"
    topn_cfg.top_n = 2
    topn_cfg.regime_filter_mode = "all"

    topn_results, topn_trades, topn_equity = backtest_scored_events(
        df=df,
        scored_events=wf_scored,
        prob_threshold=0.60,
        bt_cfg=topn_cfg,
        label_cfg=LABEL_CFG,
    )
    print_backtest_results(topn_results, "TOP-N PER DAY BACKTEST")

    # 10) top-N per day - transition only
    print("\n=== TOP-N PER DAY BACKTEST (FIXED RISK, TRANSITION ONLY) ===")
    topn_trans_cfg = BacktestConfig(**asdict(BT_CFG))
    topn_trans_cfg.position_sizing_mode = "fixed"
    topn_trans_cfg.selection_mode = "top_n_per_day"
    topn_trans_cfg.top_n = 2
    topn_trans_cfg.regime_filter_mode = "transition_only"

    topn_trans_results, topn_trans_trades, topn_trans_equity = backtest_scored_events(
        df=df,
        scored_events=wf_scored,
        prob_threshold=0.60,
        bt_cfg=topn_trans_cfg,
        label_cfg=LABEL_CFG,
    )
    print_backtest_results(topn_trans_results, "TOP-N PER DAY / TRANSITION ONLY BACKTEST")


if __name__ == "__main__":
    main()