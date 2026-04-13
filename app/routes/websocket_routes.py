"""
WebSocket routes.
Paths: WS /ws, WS /realtime/events
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websockets"])


@router.websocket("/ws")
async def websocket_ws(websocket: WebSocket):
    """WebSocket endpoint with optional JWT auth via ?token= query param."""
    token = websocket.query_params.get("token", "")
    if token:
        try:
            from app.core.auth import decode_token
            payload = decode_token(token)
            ws_user = payload.get("sub", "anonymous")
        except Exception:
            await websocket.close(code=4401, reason="Invalid token")
            return
    else:
        ws_user = "anonymous"

    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            events = [payload] if isinstance(payload, dict) else payload if isinstance(payload, list) else None
            if events is None:
                await websocket.send_json({"error": "invalid payload"})
                continue
            from app.agents.correlator import correlate_events
            from app.llm.claude import analyze_context
            correlation = correlate_events(events)
            analysis = analyze_context({"incident_id": "ws-realtime", "details": events})
            await websocket.send_json({"correlation": correlation, "analysis": analysis})
    except WebSocketDisconnect:
        pass


@router.websocket("/realtime/events")
async def websocket_events(websocket: WebSocket):
    """Real-time event correlation WebSocket."""
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            if isinstance(payload, dict):
                events = [payload]
            elif isinstance(payload, list):
                events = payload
            else:
                await websocket.send_json({"error": "invalid payload format, expected event or list"})
                continue
            from app.agents.correlator import correlate_events
            from app.llm.claude import analyze_context
            correlation = correlate_events(events)
            analysis = analyze_context({"incident_id": "realtime", "details": events})
            await websocket.send_json({"correlation": correlation, "analysis": analysis})
    except WebSocketDisconnect:
        pass
