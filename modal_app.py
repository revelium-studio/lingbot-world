"""
LingBot-World Modal Deployment

Deploys the LingBot-World backend on Modal with GPU support.
This provides serverless GPU inference for the interactive world model.

Based on: https://github.com/Robbyant/lingbot-world
"""

import os
import io
import base64
import json
import asyncio
from pathlib import Path
from typing import Optional

import modal

# Create Modal app
app = modal.App("lingbot-world")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "torchaudio",
        "diffusers>=0.31.0",
        "transformers>=4.49.0,<=4.51.3",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "opencv-python>=4.9.0.80",
        "imageio[ffmpeg]",
        "imageio-ffmpeg",
        "Pillow>=10.0.0",
        "tqdm",
        "easydict",
        "ftfy",
        "einops>=0.7.0",
        "numpy>=1.23.5,<2",
        "scipy",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.20.0",
        "fastapi>=0.109.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
    )
    .run_commands(
        # Clone LingBot-World repository
        "cd /root && git clone https://github.com/Robbyant/lingbot-world.git",
        # Remove flash_attn from dependencies (optional, requires CUDA compiler)
        "cd /root/lingbot-world && sed -i '/flash_attn/d' pyproject.toml",
        # Install as package
        "cd /root/lingbot-world && pip install -e .",
    )
)

# Create a volume to cache model weights (persists across runs)
model_volume = modal.Volume.from_name("lingbot-world-models", create_if_missing=True)
MODEL_DIR = "/models"

# GPU configuration - A100 40GB is recommended for LingBot-World
GPU_CONFIG = modal.gpu.A100(count=1, size="40GB")

# Download model weights once and cache them
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=3600,  # 1 hour for downloading
)
def download_model_weights():
    """Download LingBot-World model weights from HuggingFace."""
    from huggingface_hub import snapshot_download
    
    model_path = Path(MODEL_DIR) / "lingbot-world-base-cam"
    
    if model_path.exists() and any(model_path.glob("*.safetensors")):
        print(f"✓ Model already cached at {model_path}")
        return str(model_path)
    
    print("Downloading model weights from HuggingFace...")
    snapshot_download(
        repo_id="robbyant/lingbot-world-base-cam",
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
    )
    
    model_volume.commit()
    print(f"✓ Model downloaded to {model_path}")
    return str(model_path)


# Main inference class
@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={MODEL_DIR: model_volume},
    timeout=600,  # 10 minutes per generation
    container_idle_timeout=300,  # Keep warm for 5 minutes
    allow_concurrent_inputs=4,  # Handle multiple requests
)
class LingBotWorldModel:
    """LingBot-World model inference on Modal."""
    
    def __enter__(self):
        """Load model when container starts."""
        import sys
        import torch
        sys.path.insert(0, "/root/lingbot-world")
        
        from wan import WanI2V
        from wan.configs import WAN_CONFIGS
        
        print("Loading LingBot-World model...")
        
        model_path = Path(MODEL_DIR) / "lingbot-world-base-cam"
        self.config = WAN_CONFIGS['i2v-A14B']
        
        self.wan_i2v = WanI2V(
            config=self.config,
            checkpoint_dir=str(model_path),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,  # Keep on GPU since we have 40GB
            init_on_cpu=False,
            convert_model_dtype=False,
        )
        
        print("✓ Model loaded successfully")
    
    @modal.method()
    def generate(
        self,
        prompt: str,
        initial_image_base64: Optional[str] = None,
        intrinsics_array: Optional[list] = None,
        poses_array: Optional[list] = None,
        resolution: tuple[int, int] = (480, 832),
        num_frames: int = 161,
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        seed: int = -1,
    ) -> dict:
        """
        Generate world video from prompt and optional controls.
        
        Returns dict with base64-encoded video frames.
        """
        import tempfile
        import numpy as np
        import torch
        from PIL import Image
        from wan.configs import MAX_AREA_CONFIGS
        
        print(f"Generating: '{prompt[:50]}...' | {num_frames} frames @ {resolution}")
        
        # Process initial image
        if initial_image_base64:
            image_data = base64.b64decode(initial_image_base64)
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            # Create placeholder gradient image
            height, width = resolution
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    img_array[y, x, 0] = int(100 + 80 * x / width)
                    img_array[y, x, 1] = int(60 + 60 * y / height)
                    img_array[y, x, 2] = int(120 + 80 * (1 - x / width))
            img = Image.fromarray(img_array)
        
        # Prepare action path if camera controls provided
        action_path = None
        if intrinsics_array and poses_array:
            temp_dir = tempfile.mkdtemp()
            intrinsics = np.array(intrinsics_array, dtype=np.float32)
            poses = np.array(poses_array, dtype=np.float32)
            np.save(os.path.join(temp_dir, "intrinsics.npy"), intrinsics)
            np.save(os.path.join(temp_dir, "poses.npy"), poses)
            action_path = temp_dir
        
        # Get max_area and shift for resolution
        height, width = resolution
        size_key = f"{height}*{width}"
        max_area = MAX_AREA_CONFIGS.get(size_key, height * width)
        shift = 3.0 if height <= 480 else 5.0
        
        # Generate video
        video = self.wan_i2v.generate(
            input_prompt=prompt,
            img=img,
            action_path=action_path,
            max_area=max_area,
            frame_num=num_frames,
            shift=shift,
            sample_solver="unipc",
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=True,
        )
        
        # Convert video tensor to base64 frames
        # video shape: (C, N, H, W)
        video_cpu = video.cpu()
        video_normalized = (video_cpu + 1) / 2 * 255
        video_normalized = video_normalized.clamp(0, 255).byte()
        video_frames = video_normalized.permute(1, 2, 3, 0).numpy()  # (N, H, W, C)
        
        frames_base64 = []
        for i in range(video_frames.shape[0]):
            frame = Image.fromarray(video_frames[i])
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=85)
            frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            frames_base64.append(frame_b64)
        
        print(f"✓ Generated {len(frames_base64)} frames")
        
        return {
            "frames": frames_base64,
            "num_frames": len(frames_base64),
            "resolution": resolution,
        }


# FastAPI web endpoint
@app.function(
    image=image,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI app for HTTP/WebSocket endpoints."""
    import uuid
    import asyncio
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    
    web_app = FastAPI(title="LingBot-World API")
    
    # CORS for frontend
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # In-memory session storage
    sessions = {}
    
    # Pydantic models matching frontend expectations
    class CreateWorldRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=2000)
        initial_image_base64: Optional[str] = None
    
    class CreateWorldResponse(BaseModel):
        session_id: str
        status: str
        message: str
    
    class ControlRequest(BaseModel):
        action: str
        mouse_dx: float = 0.0
        mouse_dy: float = 0.0
    
    class ControlResponse(BaseModel):
        success: bool
        frame_index: int
    
    # Background task to generate frames
    async def generate_frames_background(session_id: str, prompt: str):
        """Generate frames in background and store in session."""
        try:
            sessions[session_id]["status"] = "generating"
            sessions[session_id]["message"] = "Generating frames on GPU..."
            
            # Call the Modal GPU function
            model = LingBotWorldModel()
            result = model.generate.remote(
                prompt=prompt,
                resolution=(480, 832),
                num_frames=81,  # Shorter for faster initial response
                sampling_steps=30,  # Faster
                guide_scale=5.0,
                seed=-1,
            )
            
            sessions[session_id]["frames"] = result.get("frames", [])
            sessions[session_id]["status"] = "ready"
            sessions[session_id]["current_frame"] = 0
            sessions[session_id]["message"] = "World generated!"
            
        except Exception as e:
            sessions[session_id]["status"] = "error"
            sessions[session_id]["message"] = str(e)
    
    # Endpoints matching frontend expectations
    @web_app.get("/")
    async def root():
        return {
            "service": "LingBot-World API",
            "status": "running",
            "model": "robbyant/lingbot-world-base-cam",
        }
    
    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    @web_app.get("/api/status")
    async def get_status():
        return {
            "model_loaded": True,
            "active_sessions": len(sessions),
            "max_sessions": 10,
            "device": "A100-40GB"
        }
    
    @web_app.post("/api/world/create", response_model=CreateWorldResponse)
    async def create_world(request: CreateWorldRequest, background_tasks: BackgroundTasks):
        """Create a new world generation session."""
        session_id = str(uuid.uuid4())
        
        # Initialize session
        sessions[session_id] = {
            "prompt": request.prompt,
            "status": "initializing",
            "message": "Starting generation...",
            "frames": [],
            "current_frame": 0,
        }
        
        # Start generation in background
        background_tasks.add_task(generate_frames_background, session_id, request.prompt)
        
        return CreateWorldResponse(
            session_id=session_id,
            status="generating",
            message="World generation started"
        )
    
    @web_app.get("/api/world/{session_id}/status")
    async def get_session_status(session_id: str):
        """Get status of a world session."""
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        return {
            "session_id": session_id,
            "prompt": session.get("prompt", ""),
            "is_generating": session.get("status") == "generating",
            "current_frame_index": session.get("current_frame", 0),
            "has_frames": len(session.get("frames", [])) > 0,
            "status": session.get("status"),
            "message": session.get("message", ""),
        }
    
    @web_app.get("/api/world/{session_id}/frame")
    async def get_current_frame(session_id: str):
        """Get the current frame as an image."""
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        frames = session.get("frames", [])
        
        if not frames:
            raise HTTPException(status_code=404, detail="No frames available yet")
        
        # Get current frame (cycle through available frames)
        frame_index = session.get("current_frame", 0) % len(frames)
        frame_b64 = frames[frame_index]
        
        # Advance frame for next request
        session["current_frame"] = (frame_index + 1) % len(frames)
        
        # Decode and return as JPEG
        import base64
        frame_data = base64.b64decode(frame_b64)
        return StreamingResponse(io.BytesIO(frame_data), media_type="image/jpeg")
    
    @web_app.post("/api/world/{session_id}/control", response_model=ControlResponse)
    async def send_control(session_id: str, request: ControlRequest):
        """Send a control action to the world."""
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        
        # For now, just advance frames based on action
        # In full implementation, this would regenerate with new camera poses
        frames = session.get("frames", [])
        if frames:
            if request.action in ["move_forward", "move_right", "turn_right"]:
                session["current_frame"] = (session.get("current_frame", 0) + 2) % len(frames)
            elif request.action in ["move_backward", "move_left", "turn_left"]:
                session["current_frame"] = (session.get("current_frame", 0) - 2) % len(frames)
        
        return ControlResponse(
            success=True,
            frame_index=session.get("current_frame", 0)
        )
    
    @web_app.delete("/api/world/{session_id}")
    async def delete_world(session_id: str):
        """End a world session."""
        if session_id in sessions:
            del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    
    @web_app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket for real-time frame streaming."""
        await websocket.accept()
        
        if session_id not in sessions:
            await websocket.close(code=4004, reason="Session not found")
            return
        
        try:
            # Send frames periodically
            frame_interval = 1.0 / 16  # 16 FPS
            
            while True:
                session = sessions.get(session_id)
                if not session:
                    break
                
                frames = session.get("frames", [])
                status = session.get("status", "unknown")
                
                if frames:
                    frame_index = session.get("current_frame", 0) % len(frames)
                    await websocket.send_json({
                        "type": "frame",
                        "frame_index": frame_index,
                        "data": frames[frame_index],
                        "is_generating": status == "generating",
                    })
                    session["current_frame"] = (frame_index + 1) % len(frames)
                else:
                    await websocket.send_json({
                        "type": "status",
                        "status": status,
                        "message": session.get("message", ""),
                    })
                
                # Check for incoming messages (non-blocking)
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=frame_interval
                    )
                    message = json.loads(data)
                    
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message.get("type") == "control":
                        # Handle control actions
                        action = message.get("action")
                        if action and frames:
                            if action in ["move_forward", "move_right", "turn_right"]:
                                session["current_frame"] = (session.get("current_frame", 0) + 2) % len(frames)
                            elif action in ["move_backward", "move_left", "turn_left"]:
                                session["current_frame"] = (session.get("current_frame", 0) - 2) % len(frames)
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(frame_interval)
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WebSocket error: {e}")
    
    return web_app


# CLI command to download weights
@app.local_entrypoint()
def download_weights():
    """Download model weights to Modal volume."""
    download_model_weights.remote()
    print("✓ Model weights downloaded and cached on Modal")
