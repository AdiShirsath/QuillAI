import asyncio
import json

import websockets


async def test_ws():
    uri = "ws://localhost:8000/ws/926bee82"
    async with websockets.connect(uri) as ws:
        while True:
            msg = await ws.recv()
            print(json.loads(msg))


asyncio.run(test_ws())
