"""
LingBot-World Interactive Application - Main Entry Point

FastAPI application that serves:
- REST API for world creation and control
- WebSocket for real-time frame streaming
- Static files for the frontend (in production)
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .api.routes import router
from .api.websocket import websocket_manager
from .session.manager import session_manager
from .model.world_generator import world_generator
from .config import settings, ensure_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting LingBot-World application...")
    ensure_directories()
    
    # Start session manager
    await session_manager.start()
    
    # Optionally pre-load model (can be slow)
    if os.getenv("PRELOAD_MODEL", "false").lower() == "true":
        logger.info("Pre-loading model...")
        await world_generator.load_model()
    
    logger.info(f"Server ready on {settings.backend_host}:{settings.backend_port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await session_manager.stop()
    world_generator.unload_model()


# Create FastAPI app
app = FastAPI(
    title="LingBot-World Interactive",
    description="Interactive world generation using LingBot-World",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        f"http://localhost:{settings.frontend_port}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API routes
app.include_router(router)


# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time communication.
    
    Connect to /ws/{session_id} after creating a world to:
    - Receive streamed frames
    - Send control commands
    """
    connected = await websocket_manager.connect(websocket, session_id)
    if not connected:
        return
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket_manager.handle_message(websocket, session_id, data)
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket, session_id)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": world_generator.is_loaded,
        "active_sessions": session_manager.get_active_session_count()
    }


# Serve frontend in production
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))


def run():
    """Run the application using uvicorn."""
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run()
