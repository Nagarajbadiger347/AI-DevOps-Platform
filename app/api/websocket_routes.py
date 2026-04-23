"""
WebSocket routes with SRE improvements: heartbeat, idle timeout, memory leak prevention.

Paths:
  WS /ws                             — event correlation + AI analysis
  WS /realtime/events                — real-time event correlation
  WS /ws/incidents/{incident_id}     — live pipeline progress stream
"""
import time
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("websocket")
router = APIRouter(tags=["websockets"])


async def websocket_with_heartbeat(
    websocket: WebSocket,
    handler,
    idle_timeout: int = 300,  # 5 minutes
    ping_interval: int = 30,   # 30 seconds
):
    """
    Generic WebSocket wrapper with heartbeat and idle timeout (SRE improvement).
    
    Args:
        websocket: FastAPI WebSocket
        handler: Async function to handle incoming messages
        idle_timeout: Close connection after N seconds of no activity
        ping_interval: Send heartbeat ping every N seconds
    """
    # from app.core.metrics import (
    #     websocket_connections_total,
    #     websocket_connections_active,
    #     websocket_disconnections_total,
    # )
    
    endpoint = websocket.url.path
    # websocket_connections_total.labels(endpoint=endpoint).inc()
    # websocket_connections_active.inc()
    
    last_activity = time.time()
    
    try:
        while True:
            # Check idle timeout
            if time.time() - last_activity > idle_timeout:
                logger.warning(f"websocket_idle_timeout path={endpoint}")
                # websocket_disconnections_total.labels(reason="idle_timeout").inc()
                await websocket.close(code=1000, reason="idle_timeout")
                break
            
            # Send periodic ping for keep-alive
            if time.time() - last_activity > ping_interval:
                try:
                    await websocket.send_json({"type": "ping"})
                    logger.debug(f"websocket_ping sent path={endpoint}")
                except Exception as e:
                    logger.error(f"websocket_ping_failed error={e}")
                    break
            
            # Receive with timeout to allow periodic ping checks
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=min(ping_interval, 10)
                )
                last_activity = time.time()
                
                # Skip ping/pong messages
                if isinstance(data, dict) and data.get("type") in ("ping", "pong"):
                    continue
                
                # Handle the actual message
                await handler(websocket, data)
                
            except asyncio.TimeoutError:
                pass  # Normal, just continue to ping interval
            except WebSocketDisconnect:
                # websocket_disconnections_total.labels(reason="client_close").inc()
                break
            except Exception as e:
                logger.error(f"websocket_handler_error error={e}", exc_info=True)
                # websocket_disconnections_total.labels(reason="error").inc()
                break
    finally:
        # websocket_connections_active.dec()
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/ws")
async def websocket_ws(websocket: WebSocket):
    """WebSocket endpoint with optional JWT auth via ?token= query param."""
    
    async def message_handler(ws: WebSocket, payload: dict):
        """Handle incoming WebSocket messages."""
        events = [payload] if isinstance(payload, dict) else (payload if isinstance(payload, list) else None)
        if events is None:
            await ws.send_json({"error": "invalid payload"})
            return
        
        try:
            from app.agents.correlator import correlate_events
            from app.llm.claude import analyze_context
            correlation = correlate_events(events)
            analysis = analyze_context({"incident_id": "ws-realtime", "details": events})
            await ws.send_json({"correlation": correlation, "analysis": analysis})
        except Exception as e:
            logger.error(f"websocket_message_handler_error error={e}")
            await ws.send_json({"error": f"Handler error: {str(e)}"})
    
    # Authenticate
    token = websocket.query_params.get("token", "")
    if token:
        try:
            from app.core.auth import decode_token
            payload = decode_token(token)
        except Exception:
            await websocket.close(code=4401, reason="Invalid token")
            return
    
    await websocket.accept()
    await websocket_with_heartbeat(websocket, message_handler)


@router.websocket("/realtime/events")
async def websocket_events(websocket: WebSocket):
    """Real-time event correlation WebSocket."""
    
    async def message_handler(ws: WebSocket, payload: dict):
        """Handle incoming events."""
        if isinstance(payload, dict):
            events = [payload]
        elif isinstance(payload, list):
            events = payload
        else:
            await ws.send_json({"error": "invalid payload format, expected event or list"})
            return
        
        from app.agents.correlator import correlate_events
        from app.llm.claude import analyze_context
        correlation = correlate_events(events)
        analysis = analyze_context({"incident_id": "realtime", "details": events})
        await websocket.send_json({"correlation": correlation, "analysis": analysis})
    
    await websocket_with_heartbeat(websocket, message_handler)


@router.websocket("/ws/incidents/{incident_id}")
async def websocket_pipeline(websocket: WebSocket, incident_id: str):
    """Push live pipeline stage events to the client for a specific incident.

    Optional auth: pass ?token=<jwt> to authenticate.
    Stream ends automatically when the pipeline finishes (status=done).
    """
    token = websocket.query_params.get("token", "")
    if token:
        try:
            from app.core.auth import decode_token
            decode_token(token)
        except Exception:
            await websocket.close(code=4401, reason="Invalid token")
            return

    await websocket.accept()

    from app.core.pipeline_events import bus

    try:
        async for event in bus.subscribe(incident_id):
            try:
                await websocket.send_json(event)
            except Exception:
                break
            if event.get("status") == "done":
                break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
