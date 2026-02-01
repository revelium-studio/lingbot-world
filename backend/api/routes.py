"""
REST API Routes for LingBot-World application.

Endpoints:
- POST /api/world/create - Create new world from prompt
- POST /api/world/{session_id}/control - Send control action
- GET /api/world/{session_id}/frame - Get current frame
- DELETE /api/world/{session_id} - End session
- GET /api/status - Server status
"""

import asyncio
import io
import base64
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from ..session.manager import session_manager
from ..model.world_generator import world_generator, GenerationConfig
from ..model.control_adapter import ControlAction
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# Request/Response Models

class CreateWorldRequest(BaseModel):
    """Request to create a new world."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    initial_image_base64: str | None = None


class CreateWorldResponse(BaseModel):
    """Response after creating a world."""
    session_id: str
    status: str
    message: str


class ControlRequest(BaseModel):
    """Request to send a control action."""
    action: ControlAction
    # Optional mouse delta for smooth camera control
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0


class ControlResponse(BaseModel):
    """Response after applying control."""
    success: bool
    frame_index: int


class SessionStatusResponse(BaseModel):
    """Status of a world session."""
    session_id: str
    prompt: str
    is_generating: bool
    current_frame_index: int
    has_frames: bool


class ServerStatusResponse(BaseModel):
    """Server status information."""
    model_loaded: bool
    active_sessions: int
    max_sessions: int
    device: str


# Endpoints

@router.get("/status", response_model=ServerStatusResponse)
async def get_server_status():
    """Get server and model status."""
    return ServerStatusResponse(
        model_loaded=world_generator.is_loaded,
        active_sessions=session_manager.get_active_session_count(),
        max_sessions=settings.max_concurrent_sessions,
        device=settings.effective_device
    )


@router.post("/world/create", response_model=CreateWorldResponse)
async def create_world(request: CreateWorldRequest):
    """
    Create a new world generation session from a text prompt.
    
    The world will begin generating immediately after creation.
    Use the WebSocket endpoint or polling to receive frames.
    """
    try:
        # Decode initial image if provided
        initial_image = None
        if request.initial_image_base64:
            try:
                image_data = base64.b64decode(request.initial_image_base64)
                initial_image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                logger.warning(f"Failed to decode initial image: {e}")
        
        # Create session
        session = await session_manager.create_session(
            prompt=request.prompt,
            initial_image=initial_image
        )
        
        # Start generation in background
        asyncio.create_task(_generate_world_background(session.session_id))
        
        return CreateWorldResponse(
            session_id=session.session_id,
            status="generating",
            message="World generation started"
        )
    
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating world: {e}")
        raise HTTPException(status_code=500, detail="Failed to create world")


async def _generate_world_background(session_id: str):
    """Background task to generate world frames."""
    session = session_manager.get_session(session_id)
    if not session:
        return
    
    session.is_generating = True
    
    try:
        config = GenerationConfig(
            prompt=session.prompt,
            initial_image=session.initial_image,
            resolution=settings.resolution,
            num_frames=settings.frame_num,
            fps=settings.default_fps
        )
        
        # Get initial camera poses from control adapter
        intrinsics, poses = session.control_adapter.get_pose_sequence(config.num_frames)
        
        async for frame in world_generator.generate_world(config, intrinsics, poses):
            # Check if session still exists
            if not session_manager.get_session(session_id):
                break
            
            session.add_frame(frame.image)
            session.current_frame_index = frame.frame_index
    
    except Exception as e:
        logger.error(f"Error generating world for session {session_id}: {e}")
    
    finally:
        if session := session_manager.get_session(session_id):
            session.is_generating = False


@router.get("/world/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get status of a world session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionStatusResponse(
        session_id=session.session_id,
        prompt=session.prompt,
        is_generating=session.is_generating,
        current_frame_index=session.current_frame_index,
        has_frames=len(session.frame_buffer) > 0
    )


@router.get("/world/{session_id}/frame")
async def get_current_frame(session_id: str, format: Literal["jpeg", "png"] = "jpeg"):
    """
    Get the current frame as an image.
    
    This is a polling endpoint for simple clients.
    For real-time streaming, use the WebSocket endpoint instead.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    frame = session.get_latest_frame()
    if not frame:
        raise HTTPException(status_code=404, detail="No frames available yet")
    
    # Encode frame
    buffer = io.BytesIO()
    frame.save(buffer, format=format.upper(), quality=85)
    buffer.seek(0)
    
    media_type = f"image/{format}"
    return StreamingResponse(buffer, media_type=media_type)


@router.post("/world/{session_id}/control", response_model=ControlResponse)
async def send_control(session_id: str, request: ControlRequest):
    """
    Send a control action to the world.
    
    This updates the camera position/rotation and triggers
    new frame generation based on the new pose.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Apply action to control adapter
    session.control_adapter.apply_action(request.action)
    
    # Apply mouse delta if provided
    if request.mouse_dx != 0 or request.mouse_dy != 0:
        session.control_adapter.apply_mouse_delta(
            request.mouse_dx,
            request.mouse_dy
        )
    
    # Trigger regeneration with new poses
    # (In full implementation, this would generate continuation frames)
    
    return ControlResponse(
        success=True,
        frame_index=session.current_frame_index
    )


@router.delete("/world/{session_id}")
async def delete_world(session_id: str):
    """End a world session and release resources."""
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "deleted", "session_id": session_id}


@router.post("/model/load")
async def load_model():
    """
    Manually trigger model loading.
    
    Useful for pre-warming the server before users arrive.
    """
    if world_generator.is_loaded:
        return {"status": "already_loaded"}
    
    try:
        await world_generator.load_model()
        return {"status": "loaded"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
