import asyncio
from dydx_v4_client.network import make_mainnet
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.indexer.candles_resolution import CandlesResolution
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESOLUTION_MINUTES = "5MINS"
MARKET = "BTC-USD"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

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


async def backfill_candles(client: IndexerClient):
    logger.info("Starting historical backfill of candles...")

    end_dt = datetime.now(timezone.utc)
    start_boundary = end_dt - timedelta(days=365)

    all_candles = []
    current_to = end_dt

    while True:
        to_iso = current_to.strftime("%Y-%m-%dT%H:%M:%SZ")

        response = await client.markets.get_perpetual_market_candles(
            market=MARKET,
            resolution=RESOLUTION_MINUTES,
            to_iso=to_iso,
            limit=1000,
        )

        candles = response.get("candles", [])
        if not candles:
            break

        all_candles.extend(candles)
        logger.info(f"Pulled {len(candles)} candles")

        # Oldest candle in this batch
        oldest_time = candles[-1]["startedAt"]
        oldest_dt = datetime.fromisoformat(
            oldest_time.replace("Z", "+00:00")
        )

        if oldest_dt <= start_boundary:
            break

        current_to = oldest_dt - timedelta(minutes=5)

        if len(candles) < 1000:
            break

    logger.info(f"Total candles collected: {len(all_candles)}")
    save_parquet(all_candles, f"{MARKET}_candles", "historical")

async def test():
    config = make_mainnet(
        node_url="dydx-grpc.publicnode.com:443",
        rest_indexer="https://indexer.dydx.trade",
        websocket_indexer="wss://indexer.dydx.trade/v4/ws"
    )

    indexer = IndexerClient(config.rest_indexer)
    await backfill_candles(indexer)

asyncio.run(test())