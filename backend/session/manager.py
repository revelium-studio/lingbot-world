"""
Session Manager: Maintains state for active world generation sessions.

Each user/browser session gets a unique world instance that persists
movement commands and generation state.
"""

import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import AsyncIterator

from PIL import Image

from ..model.control_adapter import ControlAdapter, ControlAction
from ..model.world_generator import GenerationConfig, GeneratedFrame
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class WorldSession:
    """
    Represents an active world generation session.
    
    Maintains:
    - Generation configuration (prompt, settings)
    - Camera/control state
    - Connection state for streaming
    """
    
    session_id: str
    prompt: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Control state
    control_adapter: ControlAdapter = field(default_factory=ControlAdapter)
    
    # Generation state
    is_generating: bool = False
    current_frame_index: int = 0
    
    # Optional initial image
    initial_image: Image.Image | None = None
    
    # Cached frames for smooth playback
    frame_buffer: list[Image.Image] = field(default_factory=list)
    max_buffer_size: int = 32
    
    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired due to inactivity."""
        cutoff = datetime.now() - timedelta(minutes=timeout_minutes)
        return self.last_activity < cutoff
    
    def add_frame(self, frame: Image.Image) -> None:
        """Add a frame to the buffer, removing oldest if full."""
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def get_latest_frame(self) -> Image.Image | None:
        """Get the most recent frame from the buffer."""
        return self.frame_buffer[-1] if self.frame_buffer else None


class SessionManager:
    """
    Manages multiple concurrent world sessions.
    
    Provides:
    - Session creation and cleanup
    - Lookup by session ID
    - Automatic expiration of inactive sessions
    """
    
    def __init__(self, max_sessions: int | None = None):
        self._sessions: dict[str, WorldSession] = {}
        self._max_sessions = max_sessions or settings.max_concurrent_sessions
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
    
    async def start(self) -> None:
        """Start the session manager and background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session manager started")
    
    async def stop(self) -> None:
        """Stop the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("Session manager stopped")
    
    async def create_session(
        self,
        prompt: str,
        initial_image: Image.Image | None = None
    ) -> WorldSession:
        """
        Create a new world session.
        
        Args:
            prompt: Text description of the world to generate
            initial_image: Optional starting image
            
        Returns:
            New WorldSession instance
            
        Raises:
            RuntimeError: If max sessions reached
        """
        async with self._lock:
            # Check capacity
            if len(self._sessions) >= self._max_sessions:
                # Try to clean up expired sessions first
                self._cleanup_expired()
                
                if len(self._sessions) >= self._max_sessions:
                    raise RuntimeError(
                        f"Maximum sessions ({self._max_sessions}) reached. "
                        "Please try again later."
                    )
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Create session
            session = WorldSession(
                session_id=session_id,
                prompt=prompt,
                initial_image=initial_image
            )
            
            # Initialize control adapter with correct resolution
            session.control_adapter.reset(settings.resolution)
            
            self._sessions[session_id] = session
            logger.info(f"Created session {session_id} with prompt: {prompt[:50]}...")
            
            return session
    
    def get_session(self, session_id: str) -> WorldSession | None:
        """Get a session by ID, updating its last activity."""
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                return True
            return False
    
    def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)
    
    def _cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired()
        ]
        
        for sid in expired:
            del self._sessions[sid]
            logger.info(f"Cleaned up expired session {sid}")
        
        return len(expired)
    
    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                async with self._lock:
                    count = self._cleanup_expired()
                    if count > 0:
                        logger.info(f"Cleaned up {count} expired sessions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")


# Singleton instance
session_manager = SessionManager()
