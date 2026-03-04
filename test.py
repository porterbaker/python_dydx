from dydx_v4_client.network import make_mainnet
from dydx_v4_client.node.client import NodeClient
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.indexer.socket.websocket import IndexerSocket
import asyncio

async def main():
    config = make_mainnet(
        node_url="oegs.dydx.trade:443",
        rest_indexer="https://indexer.dydx.trade",
        websocket_indexer="wss://indexer.dydx.trade/v4/ws"
    )

    node = await NodeClient.connect(config.node)

    socket = IndexerSocket(config.websocket_indexer)
    await socket.connect()

    print("Connected")

    await socket.subscribe(
        channel="v4_trades",
        id="BTC-USD"
    )

    async for message in socket:
        print(message)
        
if __name__ == "__main__":
    asyncio.run(main())