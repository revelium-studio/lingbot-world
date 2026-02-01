"""
WebSocket handler for real-time frame streaming and control.

Provides bidirectional communication:
- Server → Client: Streamed video frames as base64 JPEG
- Client → Server: Control commands (movement, camera)
"""

import asyncio
import io
import base64
import json
import logging
from dataclasses import dataclass
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image

from ..session.manager import session_manager, WorldSession
from ..model.control_adapter import ControlAction

logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Represents an active WebSocket client."""
    websocket: WebSocket
    session_id: str


class WebSocketManager:
    """
    Manages WebSocket connections and message routing.
    
    Handles:
    - Connection lifecycle
    - Frame streaming to clients
    - Control message processing
    """
    
    def __init__(self):
        self._connections: dict[str, Set[WebSocket]] = {}  # session_id -> websockets
        self._streaming_tasks: dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """
        Accept a new WebSocket connection for a session.
        
        Returns True if connection was accepted.
        """
        # Verify session exists
        session = session_manager.get_session(session_id)
        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return False
        
        await websocket.accept()
        
        # Add to connection pool
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(websocket)
        
        logger.info(f"WebSocket connected for session {session_id}")
        
        # Start streaming if not already running
        if session_id not in self._streaming_tasks:
            task = asyncio.create_task(self._stream_frames(session_id))
            self._streaming_tasks[session_id] = task
        
        return True
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket disconnection."""
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)
            
            # Stop streaming if no more connections
            if not self._connections[session_id]:
                del self._connections[session_id]
                
                if session_id in self._streaming_tasks:
                    self._streaming_tasks[session_id].cancel()
                    del self._streaming_tasks[session_id]
        
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def handle_message(self, websocket: WebSocket, session_id: str, data: str):
        """
        Process incoming WebSocket message.
        
        Expected message format:
        {
            "type": "control",
            "action": "move_forward" | "move_backward" | ...,
            "mouse_dx": 0.0,
            "mouse_dy": 0.0
        }
        """
        try:
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "control":
                await self._handle_control(session_id, message)
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {data[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_control(self, session_id: str, message: dict):
        """Handle control action from client."""
        session = session_manager.get_session(session_id)
        if not session:
            return
        
        action = message.get("action")
        if action and action in [
            "move_forward", "move_backward", "move_left", "move_right",
            "turn_left", "turn_right", "look_up", "look_down",
            "move_up", "move_down"
        ]:
            session.control_adapter.apply_action(action)
        
        # Handle mouse movement
        mouse_dx = message.get("mouse_dx", 0.0)
        mouse_dy = message.get("mouse_dy", 0.0)
        if mouse_dx != 0 or mouse_dy != 0:
            session.control_adapter.apply_mouse_delta(mouse_dx, mouse_dy)
    
    async def _stream_frames(self, session_id: str):
        """
        Stream frames to all connected clients for a session.
        
        Runs as a background task, sending frames at ~16 FPS.
        """
        frame_interval = 1.0 / 16  # 16 FPS
        last_frame_index = -1
        
        try:
            while session_id in self._connections:
                session = session_manager.get_session(session_id)
                if not session:
                    break
                
                # Get latest frame
                frame = session.get_latest_frame()
                
                if frame and session.current_frame_index != last_frame_index:
                    last_frame_index = session.current_frame_index
                    
                    # Encode frame to base64 JPEG
                    frame_data = self._encode_frame(frame)
                    
                    # Send to all connected clients
                    message = json.dumps({
                        "type": "frame",
                        "frame_index": session.current_frame_index,
                        "data": frame_data,
                        "is_generating": session.is_generating
                    })
                    
                    await self._broadcast(session_id, message)
                
                await asyncio.sleep(frame_interval)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in frame streaming: {e}")
    
    def _encode_frame(self, frame: Image.Image, quality: int = 80) -> str:
        """Encode PIL Image to base64 JPEG string."""
        buffer = io.BytesIO()
        
        # Resize if too large for efficient streaming
        max_size = (832, 480)
        if frame.size[0] > max_size[0] or frame.size[1] > max_size[1]:
            frame = frame.copy()
            frame.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        frame.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def _broadcast(self, session_id: str, message: str):
        """Send message to all clients connected to a session."""
        if session_id not in self._connections:
            return
        
        disconnected = []
        
        for websocket in self._connections[session_id]:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self._connections[session_id].discard(ws)
    
    async def send_status(self, session_id: str, status: str, message: str = ""):
        """Send a status update to all clients."""
        msg = json.dumps({
            "type": "status",
            "status": status,
            "message": message
        })
        await self._broadcast(session_id, msg)


# Singleton instance
websocket_manager = WebSocketManager()
