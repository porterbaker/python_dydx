# main_features_no_sentiment.py
import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List

import httpx
import pandas as pd
import numpy as np

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------
# Config
# ------------------------
INDEXER_HOST = "https://indexer.dydx.trade"
MARKET = "BTC-USD"
RESOLUTION = "5MINS"
LIMIT = 1000
INTERVAL = 5  # seconds for live streaming

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RESOLUTION_MINUTES = {"1MIN": 1, "5MINS": 5, "1HOUR": 60, "1DAY": 1440}

# ------------------------
# Helpers
# ------------------------
async def fetch_with_retry(client: httpx.AsyncClient, url: str, params: Dict[str, Any], max_retries=5, delay=2):
    params = {k: v for k, v in params.items() if v is not None}
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Permanent 404: {url} {params}")
                return None
            logger.warning(f"Fetch exception (attempt {attempt}/{max_retries}): {e}")
        except Exception as e:
            logger.warning(f"Fetch exception (attempt {attempt}/{max_retries}): {e}")
        await asyncio.sleep(delay * attempt)
    return None

def save_parquet(data, prefix: str, mode: str):
    """
    Save a DataFrame or list of dicts to parquet.
    """
    # Convert list of dicts to DataFrame if needed
    if isinstance(data, list):
        if not data:
            return
        df = pd.DataFrame(data)
    else:
        df = data

    path = DATA_DIR / mode
    path.mkdir(parents=True, exist_ok=True)
    filename = path / f"{prefix}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.parquet"

    # Append if file exists
    if filename.exists() and filename.stat().st_size > 0:
        existing = pd.read_parquet(filename, engine="pyarrow")
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(filename, engine="pyarrow", index=False, compression="snappy")
    logger.info(f"[{mode}] Saved {len(df)} rows to {filename}")

# ------------------------
# API Client
# ------------------------
class MarketsClient:
    def __init__(self, host: str):
        self.host = host
        self.client = httpx.AsyncClient(base_url=self.host, timeout=30)

    async def get_candles(self, market: str, resolution: str, from_iso: str, to_iso: str, limit: int = LIMIT):
        params = {"resolution": resolution, "fromISO": from_iso, "toISO": to_iso, "limit": limit}
        return await fetch_with_retry(self.client, f"/v4/candles/perpetualMarkets/{market}", params)

    async def get_funding(self, market: str, effective_before_or_at: str, limit: int = LIMIT):
        params = {"effectiveBeforeOrAt": effective_before_or_at, "limit": limit}
        return await fetch_with_retry(self.client, f"/v4/historicalFunding/{market}", params)

    async def get_orderbook(self, market: str):
        return await fetch_with_retry(self.client, f"/v4/orderbooks/perpetualMarket/{market}", {})

# ------------------------
# Feature Engineering
# ------------------------
def compute_features(candles: List[Dict], orderbook: Dict, funding: List[Dict]) -> pd.DataFrame:
    rows = []

    last_close = None
    for c in candles:
        ts = datetime.fromisoformat(c["startedAt"].replace("Z", "+00:00"))
        open_p = float(c["open"])
        high = float(c["high"])
        low = float(c["low"])
        close = float(c["close"])
        volume = float(c["volume"])
        ret = (close / last_close - 1) if last_close else 0.0
        last_close = close

        top_bid = float(orderbook.get("bids", [[0,0]])[0][0])
        top_ask = float(orderbook.get("asks", [[0,0]])[0][0])
        bid_qty = float(orderbook.get("bids", [[0,0]])[0][1])
        ask_qty = float(orderbook.get("asks", [[0,0]])[0][1])
        spread = top_ask - top_bid
        mid = (top_bid + top_ask)/2
        imbalance = (bid_qty - ask_qty)/(bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0.0
        fund_rate = float(funding[-1]["rate"]) if funding else 0.0

        rows.append({
            "timestamp": ts,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "return": ret,
            "top_bid": top_bid,
            "top_ask": top_ask,
            "spread": spread,
            "mid_price": mid,
            "orderbook_imbalance": imbalance,
            "funding_rate": fund_rate
        })

    df = pd.DataFrame(rows)
    # compute rolling features for ML
    df["ma_short"] = df["close"].rolling(3, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(10, min_periods=1).mean()
    df["volatility"] = df["close"].rolling(10, min_periods=1).std().fillna(0)
    return df

# ------------------------
# Live Streaming with Features
# ------------------------
async def live_stream_features(client: MarketsClient):
    logger.info("Starting live data streaming with ML features...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            from_iso = (now - timedelta(minutes=RESOLUTION_MINUTES[RESOLUTION])).strftime("%Y-%m-%dT%H:%M:%SZ")
            to_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

            orderbook = await client.get_orderbook(MARKET)
            candles_resp = await client.get_candles(MARKET, RESOLUTION, from_iso, to_iso)
            funding_resp = await client.get_funding(MARKET, to_iso)

            if not (candles_resp and candles_resp.get("candles") and orderbook):
                await asyncio.sleep(INTERVAL)
                continue

            df_features = compute_features(
                candles=candles_resp["candles"],
                orderbook=orderbook,
                funding=funding_resp.get("funding", [])
            )
            save_parquet(df_features, "features", "live")
            last_row = df_features.iloc[-1]
            logger.info(f"[LIVE] ts={last_row.timestamp} close={last_row.close:.2f} spread={last_row.spread:.2f} "
                        f"funding={last_row.funding_rate:.5f}")

            await asyncio.sleep(INTERVAL)
        except KeyboardInterrupt:
            logger.info("Live streaming stopped manually.")
            break

# ------------------------
# Historical Backfill
# ------------------------
async def backfill_candles(client: MarketsClient):
    logger.info("Starting historical backfill of candles...")
    to_dt = datetime.now(timezone.utc)
    start_dt = to_dt - timedelta(days=30)
    resolution_minutes = RESOLUTION_MINUTES[RESOLUTION]

    while to_dt > start_dt:
        from_dt = max(to_dt - timedelta(minutes=resolution_minutes * LIMIT), start_dt)
        to_iso = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        from_iso = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = await client.get_candles(MARKET, RESOLUTION, from_iso, to_iso)
        if data and data.get("candles"):
            save_parquet(data["candles"], f"{MARKET}_candles", "historical")
        to_dt = from_dt - timedelta(seconds=1)

async def backfill_funding(client: MarketsClient):
    logger.info("Starting historical backfill of funding...")
    to_dt = datetime.now(timezone.utc)
    start_dt = to_dt - timedelta(days=30)
    while to_dt > start_dt:
        to_iso = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = await client.get_funding(MARKET, to_iso)
        if data and data.get("funding"):
            save_parquet(data["funding"], f"{MARKET}_funding", "historical")
        to_dt -= timedelta(minutes=RESOLUTION_MINUTES[RESOLUTION] * LIMIT)

# ------------------------
# Main CLI
# ------------------------
async def main(mode: str):
    client = MarketsClient(INDEXER_HOST)
    if mode == "historical":
        await asyncio.gather(backfill_candles(client), backfill_funding(client))
    elif mode == "live":
        await live_stream_features(client)
    else:
        logger.error("Invalid mode. Choose 'historical' or 'live'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dYdX ML Data Collector")
    parser.add_argument("--mode", type=str, required=True, choices=["historical", "live"],
                        help="Mode: 'historical' or 'live'")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.mode))
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully.")