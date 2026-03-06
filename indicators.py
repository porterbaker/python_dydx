import math
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


INPUT_FILE = "./data/historical/BTC-USD_candles_2026-03-06.parquet"


# =========================================================
# CONFIG
# =========================================================

@dataclass
class StrategyConfig:
    atr_len: int = 14
    atr_mult: float = 1.4
    ema_fast: int = 20
    ema_slow: int = 50
    wick_ratio_threshold: float = 0.25
    pivot_sweep_cooldown: int = 12
    zscore_threshold: float = 1.5
    vol_spike_threshold: float = 1.2
    min_bars_trend_persistence: int = 3
    confirmation_mode: str = "close_through_mid"   # or "close_through_level"
    rv_short_window: int = 12
    rv_long_window: int = 48
    zscore_window: int = 48
    vol_window: int = 48
    oi_window: int = 48
    min_signal_score: float = 2.0


@dataclass
class LabelConfig:
    stop_atr_mult: float = 1.2
    target_atr_mult: float = 2.2
    max_holding_bars: int = 24


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.005
    stop_atr_mult: float = 1.2
    target_atr_mult: float = 2.2
    max_holding_bars: int = 24
    bars_per_year: int = 288 * 365


@dataclass
class ExecutionConfig:
    slippage_bps: float = 1.0
    entry_cooldown_bars: int = 8
    same_side_cooldown_bars: int = 16


@dataclass
class ModelConfig:
    train_frac: float = 0.60
    valid_frac: float = 0.20
    random_state: int = 42
    n_estimators: int = 400
    max_depth: int = 6
    min_samples_leaf: int = 12
    long_hours: tuple[int, ...] | None = None
    short_hours: tuple[int, ...] | None = None
    long_enabled: bool = True
    short_enabled: bool = True
    min_validation_trades: int = 15
    top_k_list: tuple[int, ...] = (5, 10, 20, 30, 50)
    min_prob_threshold: float = 0.50


# =========================================================
# DATA
# =========================================================

def load_data(input_file: str = INPUT_FILE) -> pd.DataFrame:
    df = pd.read_parquet(
        input_file,
        columns=[
            "startedAt",
            "low",
            "high",
            "open",
            "close",
            "baseTokenVolume",
            "usdVolume",
            "trades",
            "startingOpenInterest",
        ],
    ).copy()

    df["startedAt"] = pd.to_datetime(df["startedAt"])

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "baseTokenVolume",
        "usdVolume",
        "trades",
        "startingOpenInterest",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("startedAt").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

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
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(n).mean()
    return adx


def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    df = compute_atr(df, cfg.atr_len)

    # Returns / momentum
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_2"] = df["close"].pct_change(2)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)

    # Trend
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    df["ema_fast_slope"] = df["ema_fast"].diff(3)
    df["ema_slow_slope"] = df["ema_slow"].diff(5)

    # Volatility
    df["rv_short"] = df["ret_1"].rolling(cfg.rv_short_window).std()
    df["rv_long"] = df["ret_1"].rolling(cfg.rv_long_window).std()
    df["rv_ratio"] = df["rv_short"] / df["rv_long"]

    # Candle anatomy
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_range"] = candle_range
    df["body_frac"] = (df["close"] - df["open"]).abs() / candle_range
    df["upper_wick_frac"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_frac"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["close_location"] = (df["close"] - df["low"]) / candle_range

    # Mean reversion / stretch
    df["dist_ema_fast"] = (df["close"] - df["ema_fast"]) / df["ema_fast"]
    df["dist_ema_slow"] = (df["close"] - df["ema_slow"]) / df["ema_slow"]

    roll_mean = df["close"].rolling(cfg.zscore_window).mean()
    roll_std = df["close"].rolling(cfg.zscore_window).std()
    df["zscore_close"] = (df["close"] - roll_mean) / roll_std

    # Participation
    vol_med = df["usdVolume"].rolling(cfg.vol_window).median()
    trade_med = df["trades"].rolling(cfg.vol_window).median()
    df["vol_spike"] = df["usdVolume"] / vol_med
    df["trades_spike"] = df["trades"] / trade_med

    # OI / perp specific
    df["oi_delta"] = df["startingOpenInterest"].diff()
    oi_med = df["startingOpenInterest"].rolling(cfg.oi_window).median()
    df["oi_norm"] = df["startingOpenInterest"] / oi_med

    # Trend strength
    df["adx"] = rolling_adx(df, 14)

    # Time features
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

    # Bar-based candidate features not relying on sweeps
    df["high_break_20"] = (df["high"] > df["high"].shift(1).rolling(20).max()).astype(int)
    df["low_break_20"] = (df["low"] < df["low"].shift(1).rolling(20).min()).astype(int)

    # Range compression / expansion
    df["range_6"] = (df["high"].rolling(6).max() - df["low"].rolling(6).min()) / df["close"]
    df["range_24"] = (df["high"].rolling(24).max() - df["low"].rolling(24).min()) / df["close"]
    df["range_ratio"] = df["range_6"] / df["range_24"]

    # Rolling breakout distance
    rolling_high_20 = df["high"].shift(1).rolling(20).max()
    rolling_low_20 = df["low"].shift(1).rolling(20).min()
    df["dist_prev20_high"] = (df["close"] - rolling_high_20) / df["close"]
    df["dist_prev20_low"] = (df["close"] - rolling_low_20) / df["close"]

    return df


# =========================================================
# PIVOT + SWEEP ENGINE
# =========================================================

def detect_pivots_and_sweeps(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()

    df["pivot_high"] = False
    df["pivot_low"] = False
    df["pivot_price"] = np.nan

    df["sweep_high"] = False
    df["sweep_low"] = False
    df["swept_high_price"] = np.nan
    df["swept_low_price"] = np.nan
    df["sweep_midpoint"] = np.nan
    df["sweep_size_atr"] = np.nan
    df["pivot_age_bars"] = np.nan

    if len(df) <= cfg.atr_len:
        return df

    start_i = cfg.atr_len
    direction = None

    extreme_high = df.at[start_i, "high"]
    extreme_high_idx = start_i

    extreme_low = df.at[start_i, "low"]
    extreme_low_idx = start_i

    confirmed_high_pivot: tuple[int, float] | None = None
    confirmed_low_pivot: tuple[int, float] | None = None

    last_high_sweep_bar: dict[int, int] = {}
    last_low_sweep_bar: dict[int, int] = {}

    for i in range(start_i + 1, len(df)):
        high = df.at[i, "high"]
        low = df.at[i, "low"]
        close = df.at[i, "close"]
        open_ = df.at[i, "open"]
        atr = df.at[i, "atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        threshold = atr * cfg.atr_mult

        if direction is None:
            if high >= extreme_low + threshold:
                direction = "up"
                extreme_high = high
                extreme_high_idx = i
            elif low <= extreme_high - threshold:
                direction = "down"
                extreme_low = low
                extreme_low_idx = i
            else:
                if high > extreme_high:
                    extreme_high = high
                    extreme_high_idx = i
                if low < extreme_low:
                    extreme_low = low
                    extreme_low_idx = i
            continue

        if direction == "up":
            if high > extreme_high:
                extreme_high = high
                extreme_high_idx = i

            if low <= extreme_high - threshold:
                df.at[extreme_high_idx, "pivot_high"] = True
                df.at[extreme_high_idx, "pivot_price"] = extreme_high
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
                df.at[extreme_low_idx, "pivot_price"] = extreme_low
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
                i != pivot_idx
                and i - last_bar >= cfg.pivot_sweep_cooldown
                and high > pivot_price
                and close < pivot_price
                and (upper_wick / candle_range) >= cfg.wick_ratio_threshold
            ):
                df.at[i, "sweep_high"] = True
                df.at[i, "swept_high_price"] = pivot_price
                df.at[i, "sweep_midpoint"] = (high + low) / 2.0
                df.at[i, "sweep_size_atr"] = (high - pivot_price) / atr
                df.at[i, "pivot_age_bars"] = i - pivot_idx
                last_high_sweep_bar[pivot_idx] = i

        if confirmed_low_pivot is not None:
            pivot_idx, pivot_price = confirmed_low_pivot
            last_bar = last_low_sweep_bar.get(pivot_idx, -999999)

            if (
                i != pivot_idx
                and i - last_bar >= cfg.pivot_sweep_cooldown
                and low < pivot_price
                and close > pivot_price
                and (lower_wick / candle_range) >= cfg.wick_ratio_threshold
            ):
                df.at[i, "sweep_low"] = True
                df.at[i, "swept_low_price"] = pivot_price
                df.at[i, "sweep_midpoint"] = (high + low) / 2.0
                df.at[i, "sweep_size_atr"] = (pivot_price - low) / atr
                df.at[i, "pivot_age_bars"] = i - pivot_idx
                last_low_sweep_bar[pivot_idx] = i

    return df


# =========================================================
# REGIME
# =========================================================

def classify_regime(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    df["raw_regime"] = "transition"

    uptrend = (
        (df["close"] > df["ema_slow"])
        & (df["ema_fast"] > df["ema_slow"])
        & (df["ema_fast_slope"] > 0)
        & (df["ema_slow_slope"] > 0)
        & (df["adx"] > 18)
    )

    downtrend = (
        (df["close"] < df["ema_slow"])
        & (df["ema_fast"] < df["ema_slow"])
        & (df["ema_fast_slope"] < 0)
        & (df["ema_slow_slope"] < 0)
        & (df["adx"] > 18)
    )

    df.loc[uptrend, "raw_regime"] = "uptrend"
    df.loc[downtrend, "raw_regime"] = "downtrend"

    df["regime"] = "transition"
    up_streak = 0
    down_streak = 0

    for i in range(len(df)):
        r = df.at[i, "raw_regime"]

        if r == "uptrend":
            up_streak += 1
            down_streak = 0
        elif r == "downtrend":
            down_streak += 1
            up_streak = 0
        else:
            up_streak = 0
            down_streak = 0

        if up_streak >= cfg.min_bars_trend_persistence:
            df.at[i, "regime"] = "uptrend"
        elif down_streak >= cfg.min_bars_trend_persistence:
            df.at[i, "regime"] = "downtrend"
        else:
            df.at[i, "regime"] = "transition"

    return df


# =========================================================
# BASE CANDIDATE SIGNALS
# =========================================================

def build_candidate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Still builds sweep-based discretionary-style candidates for baseline research,
    but later ML uses sweeps as just one feature among many.
    """
    df = df.copy()

    df["buy"] = False
    df["sell"] = False
    df["signal_score"] = np.nan
    df["signal_side"] = None

    for i in range(1, len(df)):
        prev_i = i - 1

        prev_regime = df.at[prev_i, "regime"]
        close_now = df.at[i, "close"]
        open_now = df.at[i, "open"]

        if bool(df.at[prev_i, "sweep_low"]) and prev_regime == "uptrend":
            score = 0.0

            if abs(df.at[prev_i, "zscore_close"]) >= cfg.zscore_threshold:
                score += 1.0
            if df.at[prev_i, "vol_spike"] >= cfg.vol_spike_threshold:
                score += 1.0
            if df.at[prev_i, "lower_wick_frac"] >= cfg.wick_ratio_threshold:
                score += 1.0
            if df.at[prev_i, "close"] > df.at[prev_i, "ema_fast"]:
                score += 0.5
            if df.at[prev_i, "adx"] >= 20:
                score += 0.5
            if pd.notna(df.at[prev_i, "sweep_size_atr"]) and df.at[prev_i, "sweep_size_atr"] >= 0.20:
                score += 0.5

            confirm = False
            if cfg.confirmation_mode == "close_through_mid":
                confirm = close_now > df.at[prev_i, "sweep_midpoint"]
            elif cfg.confirmation_mode == "close_through_level":
                confirm = close_now > df.at[prev_i, "swept_low_price"]

            confirm = confirm and (close_now > open_now) and (close_now > df.at[prev_i, "high"])

            if confirm and score >= cfg.min_signal_score:
                df.at[i, "buy"] = True
                df.at[i, "signal_score"] = score
                df.at[i, "signal_side"] = "long"

        if bool(df.at[prev_i, "sweep_high"]) and prev_regime == "downtrend":
            score = 0.0

            if abs(df.at[prev_i, "zscore_close"]) >= cfg.zscore_threshold:
                score += 1.0
            if df.at[prev_i, "vol_spike"] >= cfg.vol_spike_threshold:
                score += 1.0
            if df.at[prev_i, "upper_wick_frac"] >= cfg.wick_ratio_threshold:
                score += 1.0
            if df.at[prev_i, "close"] < df.at[prev_i, "ema_fast"]:
                score += 0.5
            if df.at[prev_i, "adx"] >= 20:
                score += 0.5
            if pd.notna(df.at[prev_i, "sweep_size_atr"]) and df.at[prev_i, "sweep_size_atr"] >= 0.20:
                score += 0.5

            confirm = False
            if cfg.confirmation_mode == "close_through_mid":
                confirm = close_now < df.at[prev_i, "sweep_midpoint"]
            elif cfg.confirmation_mode == "close_through_level":
                confirm = close_now < df.at[prev_i, "swept_high_price"]

            confirm = confirm and (close_now < open_now) and (close_now < df.at[prev_i, "low"])

            if confirm and score >= cfg.min_signal_score:
                df.at[i, "sell"] = True
                df.at[i, "signal_score"] = score
                df.at[i, "signal_side"] = "short"

    both = df["buy"] & df["sell"]
    df.loc[both, "buy"] = False
    df.loc[both, "sell"] = False

    return df


def run_signal_pipeline(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = build_features(df, cfg)
    df = detect_pivots_and_sweeps(df, cfg)
    df = classify_regime(df, cfg)
    df = build_candidate_signals(df, cfg)
    return df


# =========================================================
# BROADER CANDIDATE EVENT GENERATION
# =========================================================

def build_broad_candidate_mask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates broader event candidates beyond sweeps, so ML does not depend solely on sweep logic.
    """
    df = df.copy()
    df["candidate_long"] = False
    df["candidate_short"] = False

    # Long bar-based candidates
    long_cond = (
        (
            bool(True)
            & (
                df["sweep_low"]
                | (
                    (df["regime"] == "uptrend")
                    & (df["dist_ema_fast"] < -0.002)
                    & (df["close"] > df["open"])
                )
                | (
                    (df["regime"] == "uptrend")
                    & (df["zscore_close"] < -1.0)
                    & (df["lower_wick_frac"] > 0.15)
                )
                | (
                    (df["regime"] == "uptrend")
                    & (df["low_break_20"] == 1)
                    & (df["close"] > df["open"])
                )
            )
        )
    )

    # Short bar-based candidates
    short_cond = (
        (
            bool(True)
            & (
                df["sweep_high"]
                | (
                    (df["regime"] == "downtrend")
                    & (df["dist_ema_fast"] > 0.002)
                    & (df["close"] < df["open"])
                )
                | (
                    (df["regime"] == "downtrend")
                    & (df["zscore_close"] > 1.0)
                    & (df["upper_wick_frac"] > 0.15)
                )
                | (
                    (df["regime"] == "downtrend")
                    & (df["high_break_20"] == 1)
                    & (df["close"] < df["open"])
                )
            )
        )
    )

    df.loc[long_cond.fillna(False), "candidate_long"] = True
    df.loc[short_cond.fillna(False), "candidate_short"] = True

    both = df["candidate_long"] & df["candidate_short"]
    df.loc[both, "candidate_long"] = False
    df.loc[both, "candidate_short"] = False

    return df


# =========================================================
# LABELING
# =========================================================

def triple_barrier_label(
    df: pd.DataFrame,
    event_idx: int,
    side: str,
    label_cfg: LabelConfig,
) -> tuple[int, float, int]:
    """
    Returns:
      label: +1 target first, -1 stop first, 0 time barrier first
      realized_r: realized R-multiple
      exit_idx
    """
    entry_idx = event_idx + 1
    if entry_idx >= len(df):
        return 0, np.nan, event_idx

    entry_price = df.at[entry_idx, "open"]
    atr = df.at[event_idx, "atr"]

    if pd.isna(atr) or atr <= 0:
        return 0, np.nan, event_idx

    if side == "long":
        stop_price = entry_price - label_cfg.stop_atr_mult * atr
        target_price = entry_price + label_cfg.target_atr_mult * atr
        one_r = entry_price - stop_price
    else:
        stop_price = entry_price + label_cfg.stop_atr_mult * atr
        target_price = entry_price - label_cfg.target_atr_mult * atr
        one_r = stop_price - entry_price

    last_idx = min(len(df) - 1, entry_idx + label_cfg.max_holding_bars)

    for j in range(entry_idx, last_idx + 1):
        high = df.at[j, "high"]
        low = df.at[j, "low"]

        if side == "long":
            stop_hit = low <= stop_price
            target_hit = high >= target_price

            if stop_hit and target_hit:
                return -1, -1.0, j
            if stop_hit:
                return -1, -1.0, j
            if target_hit:
                return 1, label_cfg.target_atr_mult / label_cfg.stop_atr_mult, j

        else:
            stop_hit = high >= stop_price
            target_hit = low <= target_price

            if stop_hit and target_hit:
                return -1, -1.0, j
            if stop_hit:
                return -1, -1.0, j
            if target_hit:
                return 1, label_cfg.target_atr_mult / label_cfg.stop_atr_mult, j

    exit_price = df.at[last_idx, "close"]
    if side == "long":
        r = (exit_price - entry_price) / one_r
    else:
        r = (entry_price - exit_price) / one_r

    return 0, r, last_idx


def get_model_feature_cols() -> list[str]:
    return [
        "atr",
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "ema_fast_slope",
        "ema_slow_slope",
        "rv_short",
        "rv_long",
        "rv_ratio",
        "body_frac",
        "upper_wick_frac",
        "lower_wick_frac",
        "close_location",
        "dist_ema_fast",
        "dist_ema_slow",
        "zscore_close",
        "vol_spike",
        "trades_spike",
        "oi_delta",
        "oi_norm",
        "adx",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "session_asia",
        "session_europe",
        "session_us",
        "sweep_size_atr",
        "pivot_age_bars",
        "signal_score",
        "range_6",
        "range_24",
        "range_ratio",
        "dist_prev20_high",
        "dist_prev20_low",
        "high_break_20",
        "low_break_20",
        "regime_uptrend",
        "regime_downtrend",
        "regime_transition",
        "is_sweep_event",
        "is_base_signal_event",
    ]


def build_event_dataset(
    df: pd.DataFrame,
    label_cfg: LabelConfig,
) -> pd.DataFrame:
    rows = []

    df = build_broad_candidate_mask(df)

    feature_cols = get_model_feature_cols()

    long_idx = df.index[df["candidate_long"]].tolist()
    short_idx = df.index[df["candidate_short"]].tolist()

    for event_idx, side in [(i, "long") for i in long_idx] + [(i, "short") for i in short_idx]:
        tb_label, realized_r, exit_idx = triple_barrier_label(df, event_idx, side, label_cfg)

        row = {
            "event_idx": event_idx,
            "startedAt": df.at[event_idx, "startedAt"],
            "side": side,
            "hour": int(pd.Timestamp(df.at[event_idx, "startedAt"]).hour),
            "tb_label": tb_label,
            "target_hit": int(tb_label == 1),
            "realized_r": realized_r,
            "exit_idx": exit_idx,
            "entry_regime": df.at[event_idx, "regime"],
        }

        row["regime_uptrend"] = int(df.at[event_idx, "regime"] == "uptrend")
        row["regime_downtrend"] = int(df.at[event_idx, "regime"] == "downtrend")
        row["regime_transition"] = int(df.at[event_idx, "regime"] == "transition")
        row["is_sweep_event"] = int(bool(df.at[event_idx, "sweep_low"]) or bool(df.at[event_idx, "sweep_high"]))
        row["is_base_signal_event"] = int(bool(df.at[event_idx, "buy"]) or bool(df.at[event_idx, "sell"]))

        for col in feature_cols:
            if col in row:
                continue
            row[col] = df.at[event_idx, col] if col in df.columns else np.nan

        rows.append(row)

    events = pd.DataFrame(rows).sort_values("startedAt").reset_index(drop=True)
    return events


# =========================================================
# MODELING
# =========================================================

def chronological_split(
    events: pd.DataFrame,
    train_frac: float,
    valid_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = events.sort_values("startedAt").reset_index(drop=True)

    n = len(events)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))

    train = events.iloc[:train_end].copy()
    valid = events.iloc[train_end:valid_end].copy()
    test = events.iloc[valid_end:].copy()

    return train, valid, test


def fit_side_model(
    events_side: pd.DataFrame,
    side_name: str,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    train, valid, test = chronological_split(events_side, model_cfg.train_frac, model_cfg.valid_frac)

    feature_cols = get_model_feature_cols()

    X_train = train[feature_cols]
    y_train = train["target_hit"]

    X_valid = valid[feature_cols]
    y_valid = valid["target_hit"]

    X_test = test[feature_cols]
    y_test = test["target_hit"]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp = imputer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        min_samples_leaf=model_cfg.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=model_cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X_train_imp, y_train)

    valid_prob = model.predict_proba(X_valid_imp)[:, 1] if len(valid) > 0 else np.array([])
    test_prob = model.predict_proba(X_test_imp)[:, 1] if len(test) > 0 else np.array([])

    valid_auc = roc_auc_score(y_valid, valid_prob) if len(valid) > 0 and y_valid.nunique() > 1 else np.nan
    test_auc = roc_auc_score(y_test, test_prob) if len(test) > 0 and y_test.nunique() > 1 else np.nan

    # choose threshold on validation set
    best_threshold = model_cfg.min_prob_threshold
    best_metric = -np.inf
    thresholds = np.arange(model_cfg.min_prob_threshold, 0.91, 0.02)

    if len(valid) > 0:
        valid_tmp = valid.copy()
        valid_tmp["pred_prob"] = valid_prob

        for th in thresholds:
            picked = valid_tmp[valid_tmp["pred_prob"] >= th]
            if len(picked) < model_cfg.min_validation_trades:
                continue

            metric = picked["realized_r"].mean() * np.sqrt(len(picked))
            if metric > best_metric:
                best_metric = metric
                best_threshold = float(th)

    feat_imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "side": side_name,
        "model": model,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "train_events": train,
        "valid_events": valid,
        "test_events": test,
        "valid_auc": valid_auc,
        "test_auc": test_auc,
        "threshold": best_threshold,
        "feature_importance": feat_imp,
    }


def print_model_summary(model_info: dict[str, Any]) -> None:
    print(f"\n=== {model_info['side'].upper()} MODEL ===")
    print(f"Validation AUC: {model_info['valid_auc']}")
    print(f"Test AUC:       {model_info['test_auc']}")
    print(f"Chosen threshold: {model_info['threshold']:.2f}")
    print("\nTop features:")
    print(model_info["feature_importance"].head(15).to_string(index=False))


# =========================================================
# RICHER DIAGNOSTICS
# =========================================================

def probability_decile_table(events_df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    tmp = events_df.dropna(subset=[prob_col]).copy()
    if tmp.empty:
        return pd.DataFrame()

    # rank-based qcut for robustness with repeated probs
    tmp["prob_rank"] = tmp[prob_col].rank(method="first")
    tmp["prob_decile"] = pd.qcut(tmp["prob_rank"], 10, labels=False, duplicates="drop") + 1

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


def precision_at_top_k(events_df: pd.DataFrame, prob_col: str, k_list: tuple[int, ...]) -> pd.DataFrame:
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


def cumulative_return_by_ranked_probability(events_df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    tmp = events_df.dropna(subset=[prob_col]).sort_values(prob_col, ascending=False).copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["rank"] = np.arange(1, len(tmp) + 1)
    tmp["cum_realized_r"] = tmp["realized_r"].cumsum()
    tmp["cum_hit_rate"] = tmp["target_hit"].expanding().mean()
    return tmp[["rank", prob_col, "realized_r", "cum_realized_r", "cum_hit_rate", "target_hit"]].copy()


def attach_model_probs_to_event_tables(
    long_model_info: dict[str, Any] | None,
    short_model_info: dict[str, Any] | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    long_test = None
    short_test = None

    if long_model_info is not None and len(long_model_info["test_events"]) > 0:
        test_ev = long_model_info["test_events"].copy()
        X = test_ev[long_model_info["feature_cols"]]
        X_imp = long_model_info["imputer"].transform(X)
        test_ev["pred_prob"] = long_model_info["model"].predict_proba(X_imp)[:, 1]
        long_test = test_ev

    if short_model_info is not None and len(short_model_info["test_events"]) > 0:
        test_ev = short_model_info["test_events"].copy()
        X = test_ev[short_model_info["feature_cols"]]
        X_imp = short_model_info["imputer"].transform(X)
        test_ev["pred_prob"] = short_model_info["model"].predict_proba(X_imp)[:, 1]
        short_test = test_ev

    return long_test, short_test


# =========================================================
# APPLY MODEL TO SIGNAL DF
# =========================================================

def attach_model_probs(
    df_sig: pd.DataFrame,
    events: pd.DataFrame,
    long_model_info: dict[str, Any] | None,
    short_model_info: dict[str, Any] | None,
) -> pd.DataFrame:
    df_out = df_sig.copy()
    df_out["model_prob_long"] = np.nan
    df_out["model_prob_short"] = np.nan

    for side_name, model_info in [("long", long_model_info), ("short", short_model_info)]:
        if model_info is None:
            continue

        ev = events[events["side"] == side_name].copy()
        if ev.empty:
            continue

        X = ev[model_info["feature_cols"]]
        X_imp = model_info["imputer"].transform(X)
        probs = model_info["model"].predict_proba(X_imp)[:, 1]

        prob_col = "model_prob_long" if side_name == "long" else "model_prob_short"
        for idx, prob in zip(ev["event_idx"], probs):
            df_out.at[int(idx), prob_col] = prob

    return df_out


def build_filtered_test_signals(
    df_sig: pd.DataFrame,
    long_model_info: dict[str, Any] | None,
    short_model_info: dict[str, Any] | None,
    model_cfg: ModelConfig,
) -> pd.DataFrame:
    df_out = df_sig.copy()
    df_out["buy_ml"] = False
    df_out["sell_ml"] = False

    if long_model_info is not None and model_cfg.long_enabled:
        test_ev = long_model_info["test_events"].copy()
        if len(test_ev) > 0:
            X = test_ev[long_model_info["feature_cols"]]
            X_imp = long_model_info["imputer"].transform(X)
            probs = long_model_info["model"].predict_proba(X_imp)[:, 1]
            test_ev["pred_prob"] = probs

            if model_cfg.long_hours is not None:
                test_ev = test_ev[test_ev["hour"].isin(model_cfg.long_hours)]

            test_ev = test_ev[test_ev["pred_prob"] >= long_model_info["threshold"]]

            for idx, prob in zip(test_ev["event_idx"], test_ev["pred_prob"]):
                df_out.at[int(idx), "buy_ml"] = True
                df_out.at[int(idx), "model_prob_long"] = prob

    if short_model_info is not None and model_cfg.short_enabled:
        test_ev = short_model_info["test_events"].copy()
        if len(test_ev) > 0:
            X = test_ev[short_model_info["feature_cols"]]
            X_imp = short_model_info["imputer"].transform(X)
            probs = short_model_info["model"].predict_proba(X_imp)[:, 1]
            test_ev["pred_prob"] = probs

            if model_cfg.short_hours is not None:
                test_ev = test_ev[test_ev["hour"].isin(model_cfg.short_hours)]

            test_ev = test_ev[test_ev["pred_prob"] >= short_model_info["threshold"]]

            for idx, prob in zip(test_ev["event_idx"], test_ev["pred_prob"]):
                df_out.at[int(idx), "sell_ml"] = True
                df_out.at[int(idx), "model_prob_short"] = prob

    return df_out


# =========================================================
# EXECUTION
# =========================================================

def apply_slippage(price: float, side: str, action: str, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0

    if side == "long":
        if action == "entry":
            return price * (1 + slip)
        return price * (1 - slip)
    else:
        if action == "entry":
            return price * (1 - slip)
        return price * (1 + slip)


# =========================================================
# BACKTEST
# =========================================================

def summarize_backtest(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    cfg: BacktestConfig,
) -> dict[str, Any]:
    if equity_df.empty:
        return {
            "initial_capital": cfg.initial_capital,
            "final_capital": cfg.initial_capital,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_pnl_per_trade": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "expectancy_per_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
        }

    equity_df = equity_df.copy()
    equity_df["ret"] = equity_df["equity"].pct_change().fillna(0.0)
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = equity_df["equity"] / equity_df["peak"] - 1.0

    final_capital = equity_df["equity"].iloc[-1]
    total_return_pct = (final_capital / cfg.initial_capital - 1.0) * 100.0
    max_drawdown_pct = equity_df["drawdown"].min() * 100.0

    ret_mean = equity_df["ret"].mean()
    ret_std = equity_df["ret"].std()
    downside_std = equity_df.loc[equity_df["ret"] < 0, "ret"].std()

    sharpe_ratio = 0.0
    if pd.notna(ret_std) and ret_std > 0:
        sharpe_ratio = ret_mean / ret_std * math.sqrt(cfg.bars_per_year)

    sortino_ratio = 0.0
    if pd.notna(downside_std) and downside_std > 0:
        sortino_ratio = ret_mean / downside_std * math.sqrt(cfg.bars_per_year)

    calmar_ratio = 0.0
    if max_drawdown_pct < 0:
        calmar_ratio = total_return_pct / abs(max_drawdown_pct)

    if trades_df.empty:
        return {
            "initial_capital": cfg.initial_capital,
            "final_capital": final_capital,
            "net_profit": final_capital - cfg.initial_capital,
            "total_return_pct": total_return_pct,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_pnl_per_trade": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "expectancy_per_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
        }

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.inf

    wins = trades_df.loc[trades_df["pnl"] > 0]
    losses = trades_df.loc[trades_df["pnl"] <= 0]

    win_rate_pct = (len(wins) / len(trades_df)) * 100.0
    avg_pnl_per_trade = trades_df["pnl"].mean()
    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
    payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.inf

    win_prob = len(wins) / len(trades_df)
    loss_prob = len(losses) / len(trades_df)
    expectancy_per_trade = (win_prob * avg_win) + (loss_prob * avg_loss)

    avg_bars_held = trades_df["bars_held"].mean()
    avg_return_pct = trades_df["return_pct"].mean()

    return {
        "initial_capital": cfg.initial_capital,
        "final_capital": final_capital,
        "net_profit": final_capital - cfg.initial_capital,
        "total_return_pct": total_return_pct,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "total_trades": len(trades_df),
        "win_rate_pct": win_rate_pct,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "avg_return_pct_per_trade": avg_return_pct,
        "avg_bars_held": avg_bars_held,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy_per_trade": expectancy_per_trade,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown_pct": max_drawdown_pct,
    }


def print_backtest_results(results: dict[str, Any], title: str = "BACKTEST RESULTS") -> None:
    print(f"\n=== {title} ===")
    for key, val in results.items():
        if isinstance(val, (int, np.integer)):
            print(f"{key:24s}: {val}")
        elif isinstance(val, (float, np.floating)):
            print(f"{key:24s}: {val:,.4f}")
        else:
            print(f"{key:24s}: {val}")


def backtest_strategy(
    df: pd.DataFrame,
    signal_buy_col: str,
    signal_sell_col: str,
    bt_cfg: BacktestConfig,
    ex_cfg: ExecutionConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)

    required = ["startedAt", "open", "high", "low", "close", "atr", signal_buy_col, signal_sell_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    capital = bt_cfg.initial_capital
    equity_curve = []
    trades = []

    position = None
    pending = None

    last_exit_bar = -10**9
    last_long_exit_bar = -10**9
    last_short_exit_bar = -10**9

    def size_position(entry_price: float, stop_price: float, capital_now: float) -> float:
        risk_dollars = capital_now * bt_cfg.risk_per_trade
        stop_dist = abs(entry_price - stop_price)
        if stop_dist <= 0:
            return 0.0
        return risk_dollars / stop_dist

    for i in range(len(df)):
        row = df.iloc[i]

        mtm_equity = capital
        pos_state = 0

        if position is not None:
            if position["side"] == "long":
                mtm_equity += (row["close"] - position["entry_price"]) * position["qty"]
                pos_state = 1
            else:
                mtm_equity += (position["entry_price"] - row["close"]) * position["qty"]
                pos_state = -1

        equity_curve.append({
            "startedAt": row["startedAt"],
            "equity": mtm_equity,
            "position": pos_state,
        })

        if pending is not None and position is None:
            side = pending["side"]

            if i - last_exit_bar < ex_cfg.entry_cooldown_bars:
                pending = None
            elif side == "long" and i - last_long_exit_bar < ex_cfg.same_side_cooldown_bars:
                pending = None
            elif side == "short" and i - last_short_exit_bar < ex_cfg.same_side_cooldown_bars:
                pending = None
            else:
                atr = row["atr"]
                if pd.notna(atr) and atr > 0:
                    entry_open = row["open"]
                    entry_price = apply_slippage(entry_open, side, "entry", ex_cfg.slippage_bps)

                    if side == "long":
                        stop_price = entry_price - bt_cfg.stop_atr_mult * atr
                        target_price = entry_price + bt_cfg.target_atr_mult * atr
                    else:
                        stop_price = entry_price + bt_cfg.stop_atr_mult * atr
                        target_price = entry_price - bt_cfg.target_atr_mult * atr

                    qty = size_position(entry_price, stop_price, capital)

                    if qty > 0:
                        position = {
                            "side": side,
                            "entry_time": row["startedAt"],
                            "entry_idx": i,
                            "entry_price": entry_price,
                            "entry_open_raw": entry_open,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "qty": qty,
                            "bars_held": 0,
                            "entry_atr": atr,
                            "entry_regime": pending["entry_regime"],
                            "signal_score": pending["signal_score"],
                            "model_prob": pending["model_prob"],
                            "entry_hour": pd.Timestamp(row["startedAt"]).hour,
                        }
                pending = None

        if position is not None:
            exit_reason = None
            exit_price = None

            high = row["high"]
            low = row["low"]
            close = row["close"]

            if position["side"] == "long":
                stop_hit = low <= position["stop_price"]
                target_hit = high >= position["target_price"]

                if stop_hit and target_hit:
                    exit_reason = "stop_and_target_same_bar"
                    raw_exit = position["stop_price"]
                    exit_price = apply_slippage(raw_exit, "long", "exit", ex_cfg.slippage_bps)
                elif stop_hit:
                    exit_reason = "stop"
                    raw_exit = position["stop_price"]
                    exit_price = apply_slippage(raw_exit, "long", "exit", ex_cfg.slippage_bps)
                elif target_hit:
                    exit_reason = "target"
                    raw_exit = position["target_price"]
                    exit_price = apply_slippage(raw_exit, "long", "exit", ex_cfg.slippage_bps)

            else:
                stop_hit = high >= position["stop_price"]
                target_hit = low <= position["target_price"]

                if stop_hit and target_hit:
                    exit_reason = "stop_and_target_same_bar"
                    raw_exit = position["stop_price"]
                    exit_price = apply_slippage(raw_exit, "short", "exit", ex_cfg.slippage_bps)
                elif stop_hit:
                    exit_reason = "stop"
                    raw_exit = position["stop_price"]
                    exit_price = apply_slippage(raw_exit, "short", "exit", ex_cfg.slippage_bps)
                elif target_hit:
                    exit_reason = "target"
                    raw_exit = position["target_price"]
                    exit_price = apply_slippage(raw_exit, "short", "exit", ex_cfg.slippage_bps)

            if exit_reason is None and position["bars_held"] >= bt_cfg.max_holding_bars:
                exit_reason = "time_stop"
                raw_exit = close
                exit_price = apply_slippage(raw_exit, position["side"], "exit", ex_cfg.slippage_bps)

            if exit_reason is None:
                if position["side"] == "long":
                    one_r = position["entry_price"] - position["stop_price"]
                    if high >= position["entry_price"] + one_r:
                        position["stop_price"] = max(position["stop_price"], position["entry_price"])
                else:
                    one_r = position["stop_price"] - position["entry_price"]
                    if low <= position["entry_price"] - one_r:
                        position["stop_price"] = min(position["stop_price"], position["entry_price"])

            if exit_reason is not None:
                if position["side"] == "long":
                    pnl = (exit_price - position["entry_price"]) * position["qty"]
                    ret = (exit_price / position["entry_price"]) - 1.0
                else:
                    pnl = (position["entry_price"] - exit_price) * position["qty"]
                    ret = (position["entry_price"] / exit_price) - 1.0

                capital += pnl

                trades.append({
                    "side": position["side"],
                    "entry_time": position["entry_time"],
                    "exit_time": row["startedAt"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "entry_open_raw": position["entry_open_raw"],
                    "entry_atr": position["entry_atr"],
                    "qty": position["qty"],
                    "bars_held": position["bars_held"],
                    "pnl": pnl,
                    "return_pct": ret * 100.0,
                    "exit_reason": exit_reason,
                    "entry_regime": position["entry_regime"],
                    "signal_score": position["signal_score"],
                    "model_prob": position["model_prob"],
                    "entry_hour": position["entry_hour"],
                })

                last_exit_bar = i
                if position["side"] == "long":
                    last_long_exit_bar = i
                else:
                    last_short_exit_bar = i

                position = None
            else:
                position["bars_held"] += 1

        if position is None and pending is None and i < len(df) - 1:
            buy_sig = bool(row[signal_buy_col])
            sell_sig = bool(row[signal_sell_col])

            if buy_sig and not sell_sig:
                pending = {
                    "side": "long",
                    "signal_score": row["signal_score"],
                    "entry_regime": row["regime"],
                    "model_prob": row.get("model_prob_long", np.nan),
                }
            elif sell_sig and not buy_sig:
                pending = {
                    "side": "short",
                    "signal_score": row["signal_score"],
                    "entry_regime": row["regime"],
                    "model_prob": row.get("model_prob_short", np.nan),
                }

    if position is not None:
        last_row = df.iloc[-1]
        raw_exit = last_row["close"]
        exit_price = apply_slippage(raw_exit, position["side"], "exit", ex_cfg.slippage_bps)

        if position["side"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["qty"]
            ret = (exit_price / position["entry_price"]) - 1.0
        else:
            pnl = (position["entry_price"] - exit_price) * position["qty"]
            ret = (position["entry_price"] / exit_price) - 1.0

        capital += pnl

        trades.append({
            "side": position["side"],
            "entry_time": position["entry_time"],
            "exit_time": last_row["startedAt"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "entry_open_raw": position["entry_open_raw"],
            "entry_atr": position["entry_atr"],
            "qty": position["qty"],
            "bars_held": position["bars_held"],
            "pnl": pnl,
            "return_pct": ret * 100.0,
            "exit_reason": "forced_final_exit",
            "entry_regime": position["entry_regime"],
            "signal_score": position["signal_score"],
            "model_prob": position["model_prob"],
            "entry_hour": position["entry_hour"],
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    results = summarize_backtest(trades_df, equity_df, bt_cfg)
    results["slippage_bps"] = ex_cfg.slippage_bps
    results["entry_cooldown_bars"] = ex_cfg.entry_cooldown_bars
    results["same_side_cooldown_bars"] = ex_cfg.same_side_cooldown_bars

    return results, trades_df, equity_df


# =========================================================
# REPORTING / ANALYSIS
# =========================================================

def analyze_trade_breakdown(trades_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}

    if trades_df.empty:
        return out

    out["by_side"] = (
        trades_df.groupby("side")
        .agg(
            trades=("pnl", "size"),
            net_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            win_rate=("pnl", lambda s: (s > 0).mean() * 100),
        )
        .reset_index()
    )

    if "entry_regime" in trades_df.columns:
        out["by_regime"] = (
            trades_df.groupby("entry_regime")
            .agg(
                trades=("pnl", "size"),
                net_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("pnl", lambda s: (s > 0).mean() * 100),
            )
            .reset_index()
        )

    if "entry_hour" in trades_df.columns:
        out["by_hour"] = (
            trades_df.groupby("entry_hour")
            .agg(
                trades=("pnl", "size"),
                net_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("pnl", lambda s: (s > 0).mean() * 100),
            )
            .reset_index()
        )

    if "signal_score" in trades_df.columns:
        tmp = trades_df.dropna(subset=["signal_score"]).copy()
        if not tmp.empty:
            tmp["score_bucket"] = pd.cut(
                tmp["signal_score"],
                bins=[-np.inf, 1.0, 2.0, 3.0, 4.0, np.inf],
            )
            out["by_score_bucket"] = (
                tmp.groupby("score_bucket", observed=False)
                .agg(
                    trades=("pnl", "size"),
                    net_pnl=("pnl", "sum"),
                    avg_pnl=("pnl", "mean"),
                    win_rate=("pnl", lambda s: (s > 0).mean() * 100),
                )
                .reset_index()
            )

    if "model_prob" in trades_df.columns:
        tmp = trades_df.dropna(subset=["model_prob"]).copy()
        if not tmp.empty:
            tmp["prob_bucket"] = pd.cut(
                tmp["model_prob"],
                bins=[0.0, 0.55, 0.60, 0.65, 0.70, 1.0],
                include_lowest=True,
            )
            out["by_prob_bucket"] = (
                tmp.groupby("prob_bucket", observed=False)
                .agg(
                    trades=("pnl", "size"),
                    net_pnl=("pnl", "sum"),
                    avg_pnl=("pnl", "mean"),
                    win_rate=("pnl", lambda s: (s > 0).mean() * 100),
                )
                .reset_index()
            )

    return out


def print_prob_diagnostics(title: str, events_df: pd.DataFrame, prob_col: str, k_list: tuple[int, ...]) -> None:
    if events_df is None or events_df.empty or prob_col not in events_df.columns:
        print(f"\n=== {title} ===")
        print("No events.")
        return

    print(f"\n=== {title}: PROBABILITY DECILES ===")
    dec = probability_decile_table(events_df, prob_col)
    if dec.empty:
        print("No decile data.")
    else:
        print(dec.to_string(index=False))

    print(f"\n=== {title}: PRECISION AT TOP-K ===")
    pak = precision_at_top_k(events_df, prob_col, k_list)
    if pak.empty:
        print("No top-k data.")
    else:
        print(pak.to_string(index=False))

    print(f"\n=== {title}: CUMULATIVE RETURN BY RANKED PROBABILITY (HEAD) ===")
    cum = cumulative_return_by_ranked_probability(events_df, prob_col)
    if cum.empty:
        print("No cumulative ranked-probability data.")
    else:
        print(cum.head(20).to_string(index=False))


def run_slippage_sensitivity(
    df_sig: pd.DataFrame,
    signal_buy_col: str,
    signal_sell_col: str,
    bt_cfg: BacktestConfig,
    ex_cfg: ExecutionConfig,
    slippage_bps_list: list[float] = [0.0, 0.5, 1.0, 2.0],
) -> pd.DataFrame:
    rows = []

    for s in slippage_bps_list:
        ex_tmp = ExecutionConfig(
            slippage_bps=s,
            entry_cooldown_bars=ex_cfg.entry_cooldown_bars,
            same_side_cooldown_bars=ex_cfg.same_side_cooldown_bars,
        )
        results, _, _ = backtest_strategy(df_sig, signal_buy_col, signal_sell_col, bt_cfg, ex_tmp)
        rows.append(results)

    return pd.DataFrame(rows).sort_values("slippage_bps").reset_index(drop=True)


# =========================================================
# PLOTTING
# =========================================================

def plot_signals(df: pd.DataFrame, buy_col: str, sell_col: str, tail: int | None = 500) -> None:
    df_plot = df.copy()

    if tail is not None:
        df_plot = df_plot.tail(tail).copy()

    df_plot = df_plot.set_index("startedAt")

    for col in ["open", "high", "low", "close"]:
        df_plot[col] = df_plot[col].astype(float)

    df_plot["buy_marker"] = np.where(df_plot[buy_col], df_plot["low"] * 0.995, np.nan)
    df_plot["sell_marker"] = np.where(df_plot[sell_col], df_plot["high"] * 1.005, np.nan)
    df_plot["pivot_high_marker"] = np.where(df_plot["pivot_high"], df_plot["high"] * 1.002, np.nan)
    df_plot["pivot_low_marker"] = np.where(df_plot["pivot_low"], df_plot["low"] * 0.998, np.nan)

    apds = []

    if df_plot["buy_marker"].notna().any():
        apds.append(mpf.make_addplot(df_plot["buy_marker"], type="scatter", marker="^", markersize=120))

    if df_plot["sell_marker"].notna().any():
        apds.append(mpf.make_addplot(df_plot["sell_marker"], type="scatter", marker="v", markersize=120))

    if df_plot["pivot_high_marker"].notna().any():
        apds.append(mpf.make_addplot(df_plot["pivot_high_marker"], type="scatter", marker="o", markersize=40))

    if df_plot["pivot_low_marker"].notna().any():
        apds.append(mpf.make_addplot(df_plot["pivot_low_marker"], type="scatter", marker="o", markersize=40))

    mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        addplot=apds if apds else None,
        volume=False,
        title=f"Signals: {buy_col}/{sell_col}",
        figsize=(16, 9),
    )


def plot_equity_curve(equity_df: pd.DataFrame, title: str = "Equity Curve") -> None:
    if equity_df.empty:
        return

    plt.figure(figsize=(14, 6))
    plt.plot(equity_df["startedAt"], equity_df["equity"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cumulative_ranked_probability(cum_df: pd.DataFrame, title: str) -> None:
    if cum_df is None or cum_df.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(cum_df["rank"], cum_df["cum_realized_r"])
    plt.title(title)
    plt.xlabel("Rank (highest predicted probability first)")
    plt.ylabel("Cumulative realized R")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================

def main():
    strat_cfg = StrategyConfig()
    label_cfg = LabelConfig()
    bt_cfg = BacktestConfig()
    ex_cfg = ExecutionConfig()
    model_cfg = ModelConfig()

    # 1) Full feature + signal pipeline
    df = load_data()
    df_sig = run_signal_pipeline(df, strat_cfg)

    print("\n=== BASE SIGNAL COUNTS ===")
    print("pivot highs:", int(df_sig["pivot_high"].sum()))
    print("pivot lows :", int(df_sig["pivot_low"].sum()))
    print("sweep highs:", int(df_sig["sweep_high"].sum()))
    print("sweep lows :", int(df_sig["sweep_low"].sum()))
    print("base buys  :", int(df_sig["buy"].sum()))
    print("base sells :", int(df_sig["sell"].sum()))

    # 2) Build broader event dataset (sweeps are features, not the whole event universe)
    events = build_event_dataset(df_sig, label_cfg)

    long_events = events[events["side"] == "long"].copy()
    short_events = events[events["side"] == "short"].copy()

    print("\n=== EVENT DATASET ===")
    print(f"Total events : {len(events)}")
    print(f"Long events  : {len(long_events)}")
    print(f"Short events : {len(short_events)}")
    if len(events) > 0:
        print(f"Sweep-event share: {events['is_sweep_event'].mean() * 100:.2f}%")
        print(f"Base-signal-event share: {events['is_base_signal_event'].mean() * 100:.2f}%")

    # 3) Train separate side models
    long_model_info = None
    short_model_info = None

    if model_cfg.long_enabled and len(long_events) >= 60:
        long_model_info = fit_side_model(long_events, "long", model_cfg)
        print_model_summary(long_model_info)

    if model_cfg.short_enabled and len(short_events) >= 60:
        short_model_info = fit_side_model(short_events, "short", model_cfg)
        print_model_summary(short_model_info)

    # 4) Probability diagnostics on out-of-sample test sets
    long_test_scored, short_test_scored = attach_model_probs_to_event_tables(long_model_info, short_model_info)

    print_prob_diagnostics("LONG TEST SET", long_test_scored, "pred_prob", model_cfg.top_k_list)
    print_prob_diagnostics("SHORT TEST SET", short_test_scored, "pred_prob", model_cfg.top_k_list)

    # 5) Attach probs back to bar dataframe
    df_scored = attach_model_probs(df_sig, events, long_model_info, short_model_info)

    # 6) Build OOS filtered trades using probability threshold
    df_ml = build_filtered_test_signals(df_scored, long_model_info, short_model_info, model_cfg)

    print("\n=== ML-FILTERED SIGNAL COUNTS (TEST ONLY) ===")
    print("buy_ml :", int(df_ml["buy_ml"].sum()))
    print("sell_ml:", int(df_ml["sell_ml"].sum()))

    # 7) Baseline backtest on base rule signals
    base_results, base_trades, base_equity = backtest_strategy(
        df_scored,
        signal_buy_col="buy",
        signal_sell_col="sell",
        bt_cfg=bt_cfg,
        ex_cfg=ex_cfg,
    )
    print_backtest_results(base_results, "BASE STRATEGY BACKTEST")

    # 8) OOS ML-filtered backtest
    ml_results, ml_trades, ml_equity = backtest_strategy(
        df_ml,
        signal_buy_col="buy_ml",
        signal_sell_col="sell_ml",
        bt_cfg=bt_cfg,
        ex_cfg=ex_cfg,
    )
    print_backtest_results(ml_results, "ML-FILTERED OOS BACKTEST")

    if not ml_trades.empty:
        print("\n=== ML RECENT TRADES ===")
        print(ml_trades.tail(10).to_string(index=False))

        breakdown = analyze_trade_breakdown(ml_trades)
        for name, table in breakdown.items():
            print(f"\n=== {name.upper()} ===")
            print(table.to_string(index=False))

    # 9) Slippage sensitivity on ML OOS strategy
    print("\n=== ML STRATEGY SLIPPAGE SENSITIVITY ===")
    slip_df = run_slippage_sensitivity(
        df_ml,
        signal_buy_col="buy_ml",
        signal_sell_col="sell_ml",
        bt_cfg=bt_cfg,
        ex_cfg=ex_cfg,
        slippage_bps_list=[0.0, 0.5, 1.0, 2.0],
    )
    print(
        slip_df[
            [
                "slippage_bps",
                "final_capital",
                "net_profit",
                "total_return_pct",
                "profit_factor",
                "sharpe_ratio",
                "max_drawdown_pct",
                "total_trades",
            ]
        ].to_string(index=False)
    )

    # 10) Plots
    plot_signals(df_scored, buy_col="buy", sell_col="sell", tail=500)
    plot_signals(df_ml, buy_col="buy_ml", sell_col="sell_ml", tail=500)
    plot_equity_curve(base_equity, "Base Strategy Equity")
    plot_equity_curve(ml_equity, "ML-Filtered OOS Equity")

    if long_test_scored is not None and not long_test_scored.empty:
        long_cum = cumulative_return_by_ranked_probability(long_test_scored, "pred_prob")
        plot_cumulative_ranked_probability(long_cum, "Long Test Set: Cumulative R by Ranked Probability")

    if short_test_scored is not None and not short_test_scored.empty:
        short_cum = cumulative_return_by_ranked_probability(short_test_scored, "pred_prob")
        plot_cumulative_ranked_probability(short_cum, "Short Test Set: Cumulative R by Ranked Probability")


if __name__ == "__main__":
    main()