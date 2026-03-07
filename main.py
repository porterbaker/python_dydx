import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from dydx_v4_client.network import make_mainnet
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HISTORICAL_DIR = DATA_DIR / "historical"
FEATURES_DIR = DATA_DIR / "features"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

for p in [HISTORICAL_DIR, FEATURES_DIR, SNAPSHOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

MARKET = "BTC-USD"
RESOLUTION = "5MINS"

LOOKBACK_DAYS = 545
CANDLE_LIMIT = 1000
FUNDING_LIMIT = 1000

LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================
# FILE HELPERS
# =========================================================

def utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def parquet_path(prefix: str, folder: Path) -> Path:
    return folder / f"{prefix}_{utc_today_str()}.parquet"


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def dedupe_and_sort(df: pd.DataFrame, subset: list[str], sort_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates(subset=subset, keep="last")
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def append_parquet(df_new: pd.DataFrame, path: Path, dedupe_subset: list[str], sort_cols: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        df_old = pd.read_parquet(path, engine="pyarrow")
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new.copy()

    df_all = dedupe_and_sort(df_all, subset=dedupe_subset, sort_cols=sort_cols)
    df_all.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    logger.info("Saved %s rows to %s", len(df_all), path)
    return df_all


# =========================================================
# NORMALIZERS
# =========================================================

def normalize_candles(candles: list[dict[str, Any]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "baseTokenVolume",
        "usdVolume",
        "trades",
        "startingOpenInterest",
        "orderbookMidPriceOpen",
        "orderbookMidPriceClose",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "startedAt" in df.columns:
        df["startedAt"] = safe_to_datetime(df["startedAt"])

    return df


def normalize_funding(funding_rows: list[dict[str, Any]], market: str) -> pd.DataFrame:
    """
    dYdX docs indicate historical funding is returned as response['historicalFunding'].
    This function tries to be tolerant to slight field-name variation. :contentReference[oaicite:3]{index=3}
    """
    if not funding_rows:
        return pd.DataFrame()

    df = pd.DataFrame(funding_rows)

    # Normalize timestamp column
    ts_candidates = ["effectiveAt", "effective_before_or_at", "effective_at"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"Could not find funding timestamp column in: {list(df.columns)}")

    df["effectiveAt"] = safe_to_datetime(df[ts_col])

    # Normalize funding rate column
    rate_candidates = ["rate", "fundingRate"]
    rate_col = next((c for c in rate_candidates if c in df.columns), None)
    if rate_col is None:
        raise ValueError(f"Could not find funding rate column in: {list(df.columns)}")

    df["fundingRate"] = pd.to_numeric(df[rate_col], errors="coerce")

    # Optional useful fields if present
    optional_numeric = ["price"]
    for col in optional_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "market" not in df.columns:
        df["market"] = market

    keep_cols = ["market", "effectiveAt", "fundingRate"]
    if "price" in df.columns:
        keep_cols.append("price")

    return df[keep_cols].copy()


# =========================================================
# API PULLS
# =========================================================

async def backfill_candles(client: IndexerClient, market: str, resolution: str, lookback_days: int) -> pd.DataFrame:
    logger.info("Starting historical candle backfill for %s ...", market)

    end_dt = datetime.now(timezone.utc)
    start_boundary = end_dt - timedelta(days=lookback_days)

    all_candles: list[dict[str, Any]] = []
    current_to = end_dt

    while True:
        to_iso = current_to.strftime("%Y-%m-%dT%H:%M:%SZ")

        response = await client.markets.get_perpetual_market_candles(
            market=market,
            resolution=resolution,
            to_iso=to_iso,
            limit=CANDLE_LIMIT,
        )

        candles = response.get("candles", [])
        if not candles:
            break

        all_candles.extend(candles)
        logger.info("Pulled %s candles", len(candles))

        oldest_time = candles[-1]["startedAt"]
        oldest_dt = datetime.fromisoformat(oldest_time.replace("Z", "+00:00"))

        if oldest_dt <= start_boundary:
            break

        current_to = oldest_dt - timedelta(minutes=5)

        if len(candles) < CANDLE_LIMIT:
            break

    logger.info("Total candles collected: %s", len(all_candles))
    return normalize_candles(all_candles)


async def backfill_funding(client: IndexerClient, market: str, lookback_days: int) -> pd.DataFrame:
    logger.info("Starting historical funding backfill for %s ...", market)

    end_dt = datetime.now(timezone.utc)
    start_boundary = end_dt - timedelta(days=lookback_days)

    all_funding: list[dict[str, Any]] = []
    current_before = end_dt

    while True:
        before_iso = current_before.strftime("%Y-%m-%dT%H:%M:%SZ")

        response = await client.markets.get_perpetual_market_historical_funding(
            market=market,
            effective_before_or_at=before_iso,
            limit=FUNDING_LIMIT,
        )

        funding_rows = response.get("historicalFunding", [])
        if not funding_rows:
            break

        all_funding.extend(funding_rows)
        logger.info("Pulled %s funding rows", len(funding_rows))

        oldest_ts_raw = funding_rows[-1].get("effectiveAt")
        if oldest_ts_raw is None:
            break

        oldest_dt = datetime.fromisoformat(oldest_ts_raw.replace("Z", "+00:00"))

        if oldest_dt <= start_boundary:
            break

        # funding cadence is much lower than candles; subtract 1 second to page backward safely
        current_before = oldest_dt - timedelta(seconds=1)

        if len(funding_rows) < FUNDING_LIMIT:
            break

    logger.info("Total funding rows collected: %s", len(all_funding))
    return normalize_funding(all_funding, market=market)


async def pull_market_snapshot(client: IndexerClient, market: str) -> pd.DataFrame:
    """
    markets.py exposes get_perpetual_markets(...). Saving a snapshot is useful for
    future feature engineering if the payload contains extra market metadata. :contentReference[oaicite:4]{index=4}
    """
    logger.info("Pulling market snapshot for %s ...", market)
    response = await client.markets.get_perpetual_markets(market=market)

    # API shape may vary slightly; tolerate either single-market or map-like payload
    snapshot_time = datetime.now(timezone.utc)

    if isinstance(response, dict):
        if "markets" in response and isinstance(response["markets"], dict):
            payload = response["markets"].get(market)
            if payload is not None:
                df = pd.DataFrame([payload])
            else:
                df = pd.DataFrame([response])
        else:
            df = pd.DataFrame([response])
    else:
        df = pd.DataFrame([{"raw_response": str(response)}])

    df["snapshotAt"] = snapshot_time
    df["requestedMarket"] = market
    return df


# =========================================================
# FEATURE TABLE BUILD
# =========================================================

def build_feature_table(
    candles_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    market: str,
    resolution: str,
) -> pd.DataFrame:
    if candles_df.empty:
        return pd.DataFrame()

    df = candles_df.copy()
    df = df.sort_values("startedAt").reset_index(drop=True)

    if funding_df is not None and not funding_df.empty:
        f = funding_df.copy().sort_values("effectiveAt").reset_index(drop=True)

        # Align the most recent known funding record at or before each candle start
        df = pd.merge_asof(
            df,
            f[["effectiveAt", "fundingRate"] + ([c for c in ["price"] if c in f.columns])],
            left_on="startedAt",
            right_on="effectiveAt",
            direction="backward",
        )

        # Simple funding features the model can use immediately
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df["fundingRateAbs"] = df["fundingRate"].abs()
        df["fundingRateDiff"] = df["fundingRate"].diff()
        df["fundingRateZ96"] = (
            (df["fundingRate"] - df["fundingRate"].rolling(96).mean())
            / df["fundingRate"].rolling(96).std()
        )
        df["fundingPositive"] = (df["fundingRate"] > 0).astype("Int64")
        df["fundingNegative"] = (df["fundingRate"] < 0).astype("Int64")

    # A few extra features from your existing candle schema
    if "startingOpenInterest" in df.columns:
        df["oiDelta"] = df["startingOpenInterest"].diff()
        df["oiPctChange"] = df["startingOpenInterest"].pct_change()

    if {"orderbookMidPriceOpen", "orderbookMidPriceClose"}.issubset(df.columns):
        df["midPriceReturn"] = (
            pd.to_numeric(df["orderbookMidPriceClose"], errors="coerce")
            / pd.to_numeric(df["orderbookMidPriceOpen"], errors="coerce")
            - 1.0
        )

    df["market"] = market
    df["resolution"] = resolution

    return df


# =========================================================
# MAIN
# =========================================================

async def main() -> None:
    config = make_mainnet(
        node_url="dydx-grpc.publicnode.com:443",
        rest_indexer="https://indexer.dydx.trade",
        websocket_indexer="wss://indexer.dydx.trade/v4/ws",
    )

    indexer = IndexerClient(config.rest_indexer)

    candles_df, funding_df, snapshot_df = await asyncio.gather(
        backfill_candles(indexer, MARKET, RESOLUTION, LOOKBACK_DAYS),
        backfill_funding(indexer, MARKET, LOOKBACK_DAYS),
        pull_market_snapshot(indexer, MARKET),
    )

    candles_path = parquet_path(f"{MARKET}_candles", HISTORICAL_DIR)
    funding_path = parquet_path(f"{MARKET}_funding", HISTORICAL_DIR)
    snapshot_path = parquet_path(f"{MARKET}_market_snapshot", SNAPSHOT_DIR)

    candles_df = append_parquet(
        candles_df,
        candles_path,
        dedupe_subset=["startedAt", "ticker", "resolution"] if not candles_df.empty and "ticker" in candles_df.columns else ["startedAt"],
        sort_cols=["startedAt"],
    )

    funding_df = append_parquet(
        funding_df,
        funding_path,
        dedupe_subset=["market", "effectiveAt"],
        sort_cols=["effectiveAt"],
    )

    append_parquet(
        snapshot_df,
        snapshot_path,
        dedupe_subset=["requestedMarket", "snapshotAt"],
        sort_cols=["snapshotAt"],
    )

    feature_df = build_feature_table(
        candles_df=candles_df,
        funding_df=funding_df,
        market=MARKET,
        resolution=RESOLUTION,
    )

    feature_path = parquet_path(f"{MARKET}_feature_table", FEATURES_DIR)
    append_parquet(
        feature_df,
        feature_path,
        dedupe_subset=["startedAt", "market", "resolution"],
        sort_cols=["startedAt"],
    )

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())