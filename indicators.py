import itertools
import math
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd


INPUT_FILE = "./data/historical/BTC-USD_candles_2026-03-06.parquet"


# =========================================================
# CONFIG
# =========================================================

@dataclass
class StrategyConfig:
    atr_len: int = 14
    atr_mult: float = 1.0
    ema_fast: int = 20
    ema_slow: int = 50
    wick_ratio_threshold: float = 0.10
    pivot_sweep_cooldown: int = 6
    score_threshold: float = 2.75
    rv_short_window: int = 12
    rv_long_window: int = 48
    zscore_window: int = 48
    vol_window: int = 48
    oi_window: int = 48


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.005
    stop_atr_mult: float = 1.0
    target_atr_mult: float = 1.5
    max_holding_bars: int = 24
    allow_flip: bool = True
    bars_per_year: int = 288 * 365  # 5m bars


# =========================================================
# DATA LOADING
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


def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()

    df = compute_atr(df, cfg.atr_len)

    # Returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_12"] = df["close"].pct_change(12)

    # EMAs
    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    df["ema_fast_slope"] = df["ema_fast"].diff(3)
    df["ema_slow_slope"] = df["ema_slow"].diff(5)

    # Realized vol
    df["rv_short"] = df["ret_1"].rolling(cfg.rv_short_window).std()
    df["rv_long"] = df["ret_1"].rolling(cfg.rv_long_window).std()
    df["rv_ratio"] = df["rv_short"] / df["rv_long"]

    # Candle structure
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_range"] = candle_range
    df["upper_wick_frac"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_frac"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["close_location"] = (df["close"] - df["low"]) / candle_range

    # Distance to means
    df["dist_ema_fast"] = (df["close"] - df["ema_fast"]) / df["ema_fast"]
    df["dist_ema_slow"] = (df["close"] - df["ema_slow"]) / df["ema_slow"]

    # Price z-score
    roll_mean = df["close"].rolling(cfg.zscore_window).mean()
    roll_std = df["close"].rolling(cfg.zscore_window).std()
    df["zscore_close"] = (df["close"] - roll_mean) / roll_std

    # Volume features
    if "usdVolume" in df.columns:
        vol_med = df["usdVolume"].rolling(cfg.vol_window).median()
        df["vol_spike"] = df["usdVolume"] / vol_med
    else:
        df["vol_spike"] = np.nan

    # OI features
    if "startingOpenInterest" in df.columns:
        df["oi_delta"] = df["startingOpenInterest"].diff()
        oi_med = df["startingOpenInterest"].rolling(cfg.oi_window).median()
        df["oi_norm"] = df["startingOpenInterest"] / oi_med
    else:
        df["oi_delta"] = np.nan
        df["oi_norm"] = np.nan

    return df


# =========================================================
# PIVOT + LIQUIDITY SWEEPS
# =========================================================

def detect_pivots_and_sweeps(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Streaming reversal-confirmed pivots:
    - in up leg, keep extending high until reversal confirms pivot high
    - in down leg, keep extending low until reversal confirms pivot low
    """
    df = df.copy()

    df["pivot_high"] = False
    df["pivot_low"] = False
    df["pivot_price"] = np.nan
    df["sweep_high"] = False
    df["sweep_low"] = False
    df["swept_high_price"] = np.nan
    df["swept_low_price"] = np.nan

    if len(df) <= cfg.atr_len:
        return df

    start_i = cfg.atr_len
    direction = None  # "up" / "down"

    extreme_high = df["high"].iloc[start_i]
    extreme_high_idx = start_i

    extreme_low = df["low"].iloc[start_i]
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

        # Bootstrap direction
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

        # Up leg -> confirm high on reversal
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

        # Down leg -> confirm low on reversal
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

        # Sweep above confirmed pivot high
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
                last_high_sweep_bar[pivot_idx] = i

        # Sweep below confirmed pivot low
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
                last_low_sweep_bar[pivot_idx] = i

    return df


# =========================================================
# REGIME CLASSIFICATION
# =========================================================

def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = "transition"

    uptrend = (
        (df["close"] > df["ema_slow"]) &
        (df["ema_fast"] > df["ema_slow"]) &
        (df["ema_fast_slope"] > 0) &
        (df["ema_slow_slope"] > 0)
    )

    downtrend = (
        (df["close"] < df["ema_slow"]) &
        (df["ema_fast"] < df["ema_slow"]) &
        (df["ema_fast_slope"] < 0) &
        (df["ema_slow_slope"] < 0)
    )

    range_regime = (
        (df["rv_ratio"] < 1.15) &
        (df["dist_ema_slow"].abs() < 0.01)
    )

    high_vol = df["rv_ratio"] > 1.6

    df.loc[uptrend, "regime"] = "uptrend"
    df.loc[downtrend, "regime"] = "downtrend"
    df.loc[range_regime, "regime"] = "range"
    df.loc[high_vol, "regime"] = "high_vol"

    return df


# =========================================================
# EVENT SCORING
# =========================================================

def score_events(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()

    df["long_score"] = 0.0
    df["short_score"] = 0.0

    # Base sweep
    df.loc[df["sweep_low"], "long_score"] += 1.25
    df.loc[df["sweep_high"], "short_score"] += 1.25

    # Regime alignment
    df.loc[df["regime"] == "uptrend", "long_score"] += 1.0
    df.loc[df["regime"] == "downtrend", "short_score"] += 1.0

    # Secondary support: range mean reversion
    df.loc[(df["regime"] == "range") & df["sweep_low"], "long_score"] += 0.35
    df.loc[(df["regime"] == "range") & df["sweep_high"], "short_score"] += 0.35

    # Wick rejection
    df.loc[df["lower_wick_frac"] > 0.20, "long_score"] += 0.75
    df.loc[df["upper_wick_frac"] > 0.20, "short_score"] += 0.75

    # Stretch / dislocation
    df.loc[df["zscore_close"] < -1.0, "long_score"] += 0.75
    df.loc[df["zscore_close"] > 1.0, "short_score"] += 0.75

    # Distance from fast mean
    df.loc[df["dist_ema_fast"] < -0.0025, "long_score"] += 0.50
    df.loc[df["dist_ema_fast"] > 0.0025, "short_score"] += 0.50

    # Volume confirmation
    if "vol_spike" in df.columns:
        df.loc[df["vol_spike"] > 1.2, "long_score"] += 0.40
        df.loc[df["vol_spike"] > 1.2, "short_score"] += 0.40

    # OI flush / unwind can support mean reversion
    if "oi_delta" in df.columns:
        df.loc[df["oi_delta"] < 0, "long_score"] += 0.30
        df.loc[df["oi_delta"] < 0, "short_score"] += 0.30

    # Avoid trading unstable transitions too aggressively
    df.loc[df["regime"] == "transition", "long_score"] -= 0.50
    df.loc[df["regime"] == "transition", "short_score"] -= 0.50

    df["buy"] = df["long_score"] >= cfg.score_threshold
    df["sell"] = df["short_score"] >= cfg.score_threshold

    # No simultaneous buy/sell
    both = df["buy"] & df["sell"]
    df.loc[both, "buy"] = False
    df.loc[both, "sell"] = False

    return df


# =========================================================
# PIPELINE
# =========================================================

def run_signal_pipeline(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = build_features(df, cfg)
    df = detect_pivots_and_sweeps(df, cfg)
    df = classify_regime(df)
    df = score_events(df, cfg)
    return df


# =========================================================
# BACKTEST
# =========================================================

def backtest_strategy(
    df: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)

    required = ["startedAt", "open", "high", "low", "close", "buy", "sell", "atr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    capital = cfg.initial_capital
    equity_curve = []
    trades = []

    position = None
    pending = None

    def size_position(entry_price: float, stop_price: float, capital_now: float) -> float:
        risk_dollars = capital_now * cfg.risk_per_trade
        stop_dist = abs(entry_price - stop_price)
        if stop_dist <= 0:
            return 0.0
        return risk_dollars / stop_dist

    for i in range(len(df)):
        row = df.iloc[i]

        # Mark-to-market equity
        mtm_equity = capital
        position_state = 0
        if position is not None:
            if position["side"] == "long":
                mtm_equity += (row["close"] - position["entry_price"]) * position["qty"]
                position_state = 1
            else:
                mtm_equity += (position["entry_price"] - row["close"]) * position["qty"]
                position_state = -1

        equity_curve.append({
            "startedAt": row["startedAt"],
            "equity": mtm_equity,
            "position": position_state,
        })

        # Enter at current bar open if pending
        if pending is not None and position is None:
            atr = row["atr"]
            if pd.notna(atr) and atr > 0:
                entry_price = row["open"]

                if pending["side"] == "long":
                    stop_price = entry_price - cfg.stop_atr_mult * atr
                    target_price = entry_price + cfg.target_atr_mult * atr
                else:
                    stop_price = entry_price + cfg.stop_atr_mult * atr
                    target_price = entry_price - cfg.target_atr_mult * atr

                qty = size_position(entry_price, stop_price, capital)

                if qty > 0:
                    position = {
                        "side": pending["side"],
                        "entry_time": row["startedAt"],
                        "entry_idx": i,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "qty": qty,
                        "bars_held": 0,
                        "entry_atr": atr,
                    }
            pending = None

        # Manage position
        if position is not None:
            exit_reason = None
            exit_price = None

            high = row["high"]
            low = row["low"]

            if position["side"] == "long":
                stop_hit = low <= position["stop_price"]
                target_hit = high >= position["target_price"]

                # conservative same-bar handling
                if stop_hit and target_hit:
                    exit_reason = "stop_and_target_same_bar"
                    exit_price = position["stop_price"]
                elif stop_hit:
                    exit_reason = "stop"
                    exit_price = position["stop_price"]
                elif target_hit:
                    exit_reason = "target"
                    exit_price = position["target_price"]

            else:
                stop_hit = high >= position["stop_price"]
                target_hit = low <= position["target_price"]

                if stop_hit and target_hit:
                    exit_reason = "stop_and_target_same_bar"
                    exit_price = position["stop_price"]
                elif stop_hit:
                    exit_reason = "stop"
                    exit_price = position["stop_price"]
                elif target_hit:
                    exit_reason = "target"
                    exit_price = position["target_price"]

            # Opposite signal exit if still open
            if exit_reason is None:
                if position["side"] == "long" and row["sell"]:
                    exit_reason = "opposite_signal"
                    exit_price = row["close"]
                elif position["side"] == "short" and row["buy"]:
                    exit_reason = "opposite_signal"
                    exit_price = row["close"]

            # Time stop
            if exit_reason is None and position["bars_held"] >= cfg.max_holding_bars:
                exit_reason = "time_stop"
                exit_price = row["close"]

            # Trail stop to breakeven after 1R
            if exit_reason is None:
                if position["side"] == "long":
                    one_r = position["entry_price"] - position["stop_price"]
                    if high >= position["entry_price"] + one_r:
                        position["stop_price"] = max(position["stop_price"], position["entry_price"])
                else:
                    one_r = position["stop_price"] - position["entry_price"]
                    if low <= position["entry_price"] - one_r:
                        position["stop_price"] = min(position["stop_price"], position["entry_price"])

            # Close trade
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
                    "entry_atr": position["entry_atr"],
                    "qty": position["qty"],
                    "bars_held": position["bars_held"],
                    "pnl": pnl,
                    "return_pct": ret * 100.0,
                    "r_multiple": pnl / (cfg.initial_capital * cfg.risk_per_trade) if cfg.initial_capital > 0 else np.nan,
                    "exit_reason": exit_reason,
                })

                if cfg.allow_flip:
                    if row["buy"] and position["side"] == "short":
                        pending = {"side": "long"}
                    elif row["sell"] and position["side"] == "long":
                        pending = {"side": "short"}

                position = None
            else:
                position["bars_held"] += 1

        # Create new entry order for next bar
        if position is None and pending is None and i < len(df) - 1:
            if row["buy"] and not row["sell"]:
                pending = {"side": "long"}
            elif row["sell"] and not row["buy"]:
                pending = {"side": "short"}

    # Force close final open position
    if position is not None:
        last_row = df.iloc[-1]
        exit_price = last_row["close"]

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
            "entry_atr": position["entry_atr"],
            "qty": position["qty"],
            "bars_held": position["bars_held"],
            "pnl": pnl,
            "return_pct": ret * 100.0,
            "r_multiple": pnl / (cfg.initial_capital * cfg.risk_per_trade) if cfg.initial_capital > 0 else np.nan,
            "exit_reason": "forced_final_exit",
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    results = summarize_backtest(trades_df, equity_df, cfg)

    return results, trades_df, equity_df


# =========================================================
# METRICS
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


def print_backtest_results(results: dict[str, Any]) -> None:
    print("\n=== BACKTEST RESULTS ===")
    for key, val in results.items():
        if isinstance(val, (int, np.integer)):
            print(f"{key:24s}: {val}")
        elif isinstance(val, (float, np.floating)):
            print(f"{key:24s}: {val:,.4f}")
        else:
            print(f"{key:24s}: {val}")


# =========================================================
# PARAMETER SWEEP
# =========================================================

def parameter_sweep(
    df_raw: pd.DataFrame,
    strategy_grid: dict[str, list[Any]],
    backtest_grid: dict[str, list[Any]],
) -> pd.DataFrame:
    rows = []

    strategy_keys = list(strategy_grid.keys())
    backtest_keys = list(backtest_grid.keys())

    strategy_vals = [strategy_grid[k] for k in strategy_keys]
    backtest_vals = [backtest_grid[k] for k in backtest_keys]

    for strat_combo in itertools.product(*strategy_vals):
        strat_kwargs = dict(zip(strategy_keys, strat_combo))
        strat_cfg = StrategyConfig(**strat_kwargs)

        df_sig = run_signal_pipeline(df_raw.copy(), strat_cfg)

        for bt_combo in itertools.product(*backtest_vals):
            bt_kwargs = dict(zip(backtest_keys, bt_combo))
            bt_cfg = BacktestConfig(**bt_kwargs)

            results, trades_df, equity_df = backtest_strategy(df_sig, bt_cfg)

            row = {
                **strat_kwargs,
                **bt_kwargs,
                **results,
            }
            rows.append(row)

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        results_df = results_df.sort_values(
            by=["sharpe_ratio", "profit_factor", "total_return_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    return results_df


# =========================================================
# PLOTTING
# =========================================================

def plot_signals(df: pd.DataFrame, tail: int | None = 500) -> None:
    df_plot = df.copy()

    if tail is not None:
        df_plot = df_plot.tail(tail).copy()

    df_plot = df_plot.set_index("startedAt")

    for col in ["open", "high", "low", "close"]:
        df_plot[col] = df_plot[col].astype(float)

    df_plot["buy_marker"] = np.where(df_plot["buy"], df_plot["low"] * 0.995, np.nan)
    df_plot["sell_marker"] = np.where(df_plot["sell"], df_plot["high"] * 1.005, np.nan)
    df_plot["pivot_high_marker"] = np.where(df_plot["pivot_high"], df_plot["high"] * 1.002, np.nan)
    df_plot["pivot_low_marker"] = np.where(df_plot["pivot_low"], df_plot["low"] * 0.998, np.nan)

    apds = [
        mpf.make_addplot(df_plot["buy_marker"], type="scatter", marker="^", markersize=120),
        mpf.make_addplot(df_plot["sell_marker"], type="scatter", marker="v", markersize=120),
        mpf.make_addplot(df_plot["pivot_high_marker"], type="scatter", marker="o", markersize=45),
        mpf.make_addplot(df_plot["pivot_low_marker"], type="scatter", marker="o", markersize=45),
    ]

    mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        addplot=apds,
        volume=False,
        title="BTC Research Pipeline Signals",
        figsize=(16, 9),
    )


def plot_equity_curve(equity_df: pd.DataFrame) -> None:
    if equity_df.empty:
        print("No equity data to plot.")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(equity_df["startedAt"], equity_df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# EVENT STUDY
# =========================================================

def event_study(df: pd.DataFrame, horizons: list[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
    df = df.copy()

    rows = []
    long_events = df.index[df["buy"]].tolist()
    short_events = df.index[df["sell"]].tolist()

    for side, events in [("long", long_events), ("short", short_events)]:
        for h in horizons:
            rets = []
            for i in events:
                if i + h >= len(df):
                    continue

                entry = df.at[i, "close"]
                exit_ = df.at[i + h, "close"]

                if side == "long":
                    r = exit_ / entry - 1.0
                else:
                    r = entry / exit_ - 1.0

                rets.append(r)

            rows.append({
                "side": side,
                "horizon_bars": h,
                "n_events": len(rets),
                "mean_return_pct": np.mean(rets) * 100.0 if rets else np.nan,
                "median_return_pct": np.median(rets) * 100.0 if rets else np.nan,
                "hit_rate_pct": (np.mean(np.array(rets) > 0) * 100.0) if rets else np.nan,
            })

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================

def main():
    df = load_data()

    # -----------------------------
    # Base run
    # -----------------------------
    strat_cfg = StrategyConfig(
        atr_len=14,
        atr_mult=0.9,
        ema_fast=20,
        ema_slow=50,
        wick_ratio_threshold=0.10,
        pivot_sweep_cooldown=5,
        score_threshold=2.75,
        rv_short_window=12,
        rv_long_window=48,
        zscore_window=48,
        vol_window=48,
        oi_window=48,
    )

    bt_cfg = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.005,
        stop_atr_mult=1.0,
        target_atr_mult=1.6,
        max_holding_bars=24,
        allow_flip=True,
    )

    df_sig = run_signal_pipeline(df, strat_cfg)
    results, trades_df, equity_df = backtest_strategy(df_sig, bt_cfg)

    print_backtest_results(results)

    print("\n=== SIGNAL COUNTS ===")
    print("pivot highs:", int(df_sig["pivot_high"].sum()))
    print("pivot lows :", int(df_sig["pivot_low"].sum()))
    print("sweep highs:", int(df_sig["sweep_high"].sum()))
    print("sweep lows :", int(df_sig["sweep_low"].sum()))
    print("buys       :", int(df_sig["buy"].sum()))
    print("sells      :", int(df_sig["sell"].sum()))

    if not trades_df.empty:
        print("\n=== RECENT TRADES ===")
        print(trades_df.tail(10).to_string(index=False))

    print("\n=== EVENT STUDY ===")
    print(event_study(df_sig).to_string(index=False))

    plot_signals(df_sig, tail=500)
    plot_equity_curve(equity_df)

    # -----------------------------
    # Example parameter sweep
    # -----------------------------
    run_grid_search = False

    if run_grid_search:
        strategy_grid = {
            "atr_len": [14],
            "atr_mult": [0.8, 0.9, 1.0, 1.2],
            "ema_fast": [10, 20],
            "ema_slow": [30, 50],
            "wick_ratio_threshold": [0.05, 0.10, 0.15],
            "pivot_sweep_cooldown": [4, 6, 8],
            "score_threshold": [2.5, 2.75, 3.0],
            "rv_short_window": [12],
            "rv_long_window": [48],
            "zscore_window": [48],
            "vol_window": [48],
            "oi_window": [48],
        }

        backtest_grid = {
            "initial_capital": [10_000.0],
            "risk_per_trade": [0.003, 0.005],
            "stop_atr_mult": [0.8, 1.0, 1.2],
            "target_atr_mult": [1.2, 1.6, 2.0],
            "max_holding_bars": [12, 24, 36],
            "allow_flip": [True],
        }

        sweep_df = parameter_sweep(df, strategy_grid, backtest_grid)

        print("\n=== TOP PARAMETER SETS ===")
        print(
            sweep_df[
                [
                    "atr_mult",
                    "ema_fast",
                    "ema_slow",
                    "wick_ratio_threshold",
                    "pivot_sweep_cooldown",
                    "score_threshold",
                    "stop_atr_mult",
                    "target_atr_mult",
                    "max_holding_bars",
                    "total_return_pct",
                    "sharpe_ratio",
                    "profit_factor",
                    "max_drawdown_pct",
                    "total_trades",
                    "win_rate_pct",
                ]
            ].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()