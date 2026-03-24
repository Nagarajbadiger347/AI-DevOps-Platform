import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://127.0.0.1:8000/realtime/events"
    async with websockets.connect(uri) as websocket:
        # Send a test event
        test_event = {
            "id": "ws-test-1",
            "type": "error",
            "source": "test",
            "payload": {"message": "WebSocket test event"}
        }
        await websocket.send(json.dumps(test_event))
        print(f"Sent: {test_event}")

        # Receive response
        response = await websocket.recv()
        print(f"Received: {json.loads(response)}")

        # Send another event (list)
        test_events = [
            {"id": "ws-test-2", "type": "warning", "source": "aws", "payload": {"message": "EC2 high CPU"}},
            {"id": "ws-test-3", "type": "error", "source": "aws", "payload": {"message": "S3 access denied"}}
        ]
        await websocket.send(json.dumps(test_events))
        print(f"Sent: {test_events}")

        # Receive response
        response = await websocket.recv()
        print(f"Received: {json.loads(response)}")

if __name__ == "__main__":
    asyncio.run(test_websocket())