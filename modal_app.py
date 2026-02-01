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
# Using string "A100-40GB" in class decorator

# Download model weights once and cache them
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=7200,  # 2 hours for downloading large files
)
def download_model_weights(force: bool = False):
    """Download LingBot-World model weights from HuggingFace."""
    import subprocess
    import shutil
    
    model_path = Path(MODEL_DIR) / "lingbot-world-base-cam"
    
    print("=" * 60)
    print("LingBot-World Model Downloader")
    print("=" * 60)
    
    # List existing files in volume
    print(f"\nChecking volume at {MODEL_DIR}...")
    if Path(MODEL_DIR).exists():
        all_files = list(Path(MODEL_DIR).rglob("*"))
        print(f"Total items in volume: {len(all_files)}")
        
        # Check for safetensors files specifically
        safetensor_files = list(Path(MODEL_DIR).rglob("*.safetensors"))
        print(f"Safetensor files found: {len(safetensor_files)}")
        for f in safetensor_files[:5]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name}: {size_mb:.1f} MB")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())
        print(f"Total size: {total_size / (1024**3):.2f} GB")
    else:
        print("Volume is empty")
    
    # Check if model is already properly downloaded (should be ~50GB+)
    if model_path.exists() and not force:
        safetensors = list(model_path.glob("*.safetensors"))
        if safetensors:
            total_size = sum(f.stat().st_size for f in safetensors)
            if total_size > 10 * 1024**3:  # More than 10GB
                print(f"\n✓ Model already cached at {model_path}")
                print(f"  Total safetensor size: {total_size / (1024**3):.1f} GB")
                return str(model_path)
            else:
                print(f"\n⚠️ Model files incomplete ({total_size / (1024**3):.2f} GB)")
                print("  Re-downloading...")
                shutil.rmtree(model_path, ignore_errors=True)
    
    # Use huggingface-cli for robust downloading with progress
    print("\n" + "=" * 60)
    print("Downloading model from HuggingFace...")
    print("This may take 30-60 minutes for the full ~50GB model")
    print("=" * 60 + "\n")
    
    # Create directory
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Download using huggingface_hub with resume support
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id="robbyant/lingbot-world-base-cam",
        local_dir=str(model_path),
        resume_download=True,
        max_workers=4,
    )
    
    # Verify download
    print("\n" + "=" * 60)
    print("Verifying download...")
    safetensors = list(model_path.glob("*.safetensors"))
    total_size = sum(f.stat().st_size for f in safetensors) if safetensors else 0
    
    print(f"Safetensor files: {len(safetensors)}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    
    for f in sorted(safetensors, key=lambda x: x.stat().st_size, reverse=True)[:10]:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")
    
    if total_size < 10 * 1024**3:
        print("\n❌ WARNING: Download may be incomplete!")
        print("   Expected ~50GB, got {:.2f} GB".format(total_size / (1024**3)))
    else:
        print("\n✓ Download complete!")
    
    # Commit changes to volume
    model_volume.commit()
    print("✓ Volume committed")
    print("=" * 60)
    
    return str(model_path)


# Main inference class
@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_DIR: model_volume},
    timeout=600,  # 10 minutes per generation
    scaledown_window=300,  # Keep warm for 5 minutes
)
class LingBotWorldModel:
    """LingBot-World model inference on Modal."""
    
    @modal.enter()
    def load_model(self):
        """Load model when container starts."""
        import sys
        import torch
        sys.path.insert(0, "/root/lingbot-world")
        
        print("=" * 60)
        print("Initializing LingBot-World model...")
        print("=" * 60)
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        model_path = Path(MODEL_DIR) / "lingbot-world-base-cam"
        print(f"\nModel path: {model_path}")
        print(f"Model path exists: {model_path.exists()}")
        
        if model_path.exists():
            # List top-level contents
            top_files = list(model_path.glob("*"))
            print(f"\nTop-level items: {len(top_files)}")
            for f in top_files:
                if f.is_dir():
                    subfiles = list(f.glob("*"))
                    print(f"  📁 {f.name}/ ({len(subfiles)} files)")
                else:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  📄 {f.name} ({size_mb:.1f} MB)")
            
            # Check for safetensors recursively
            safetensors = list(model_path.rglob("*.safetensors"))
            print(f"\nTotal safetensor files (recursive): {len(safetensors)}")
            if safetensors:
                total_size = sum(f.stat().st_size for f in safetensors)
                print(f"Total safetensor size: {total_size / (1024**3):.1f} GB")
        
        # Check if model weights exist - use rglob for recursive search
        safetensor_files = list(model_path.rglob("*.safetensors")) if model_path.exists() else []
        t5_file = model_path / "models_t5_umt5-xxl-enc-bf16.pth"
        
        if not safetensor_files or not t5_file.exists():
            print("\n⚠️ Model weights not found! Running in demo mode.")
            print("Missing safetensors:", len(safetensor_files) == 0)
            print("Missing T5 file:", not t5_file.exists())
            print("Run: python3 -m modal run modal_app.py::download_weights")
            self.wan_i2v = None
            self.demo_mode = True
            return
        
        print("\n✓ Model files verified!")
        print("Loading WanI2V model (this may take 1-2 minutes)...")
        
        try:
            from wan import WanI2V
            from wan.configs import WAN_CONFIGS
            
            self.config = WAN_CONFIGS['i2v-A14B']
            print(f"Config: i2v-A14B")
            
            self.wan_i2v = WanI2V(
                config=self.config,
                checkpoint_dir=str(model_path),
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=True,  # Move T5 to CPU to save GPU memory
                init_on_cpu=False,
                convert_model_dtype=False,
            )
            self.demo_mode = False
            print("\n" + "=" * 60)
            print("✓ MODEL LOADED SUCCESSFULLY!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.wan_i2v = None
            self.demo_mode = True
        
        print("=" * 60)
    
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
        
        print(f"Generating: '{prompt[:50]}...' | {num_frames} frames @ {resolution}")
        
        # Check if in demo mode
        if getattr(self, 'demo_mode', True) or self.wan_i2v is None:
            print("Running in DEMO MODE - generating placeholder frames")
            return self._generate_demo_frames(prompt, resolution, num_frames)
        
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
        from wan.configs import MAX_AREA_CONFIGS
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
    
    def _generate_demo_frames(
        self,
        prompt: str,
        resolution: tuple[int, int],
        num_frames: int
    ) -> dict:
        """Generate demo placeholder frames when model isn't available."""
        import numpy as np
        from PIL import Image
        import hashlib
        
        height, width = resolution
        frames_base64 = []
        
        # Use prompt hash for consistent colors
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        np.random.seed(prompt_hash % (2**31))
        base_color = [np.random.randint(40, 180) for _ in range(3)]
        
        print(f"Generating {num_frames} demo frames...")
        
        for i in range(min(num_frames, 60)):  # Cap at 60 for demo
            # Create animated gradient
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            t = i / max(num_frames, 1)
            
            for c in range(3):
                base = base_color[c] + int(40 * np.sin(t * 2 * np.pi + c * 2.1))
                img_array[:, :, c] = np.clip(base, 0, 255)
            
            # Add gradient
            for y in range(height):
                gradient = int(40 * y / height)
                img_array[y, :, :] = np.clip(
                    img_array[y, :, :].astype(np.int16) + gradient, 0, 255
                ).astype(np.uint8)
            
            # Add some noise
            noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
            img_array = np.clip(
                img_array.astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)
            
            # Encode to base64
            frame = Image.fromarray(img_array)
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=85)
            frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            frames_base64.append(frame_b64)
        
        print(f"✓ Generated {len(frames_base64)} demo frames")
        
        return {
            "frames": frames_base64,
            "num_frames": len(frames_base64),
            "resolution": resolution,
            "demo_mode": True,
        }


# FastAPI web endpoint
@app.function(
    image=image,
    min_containers=1,  # Keep one instance warm to avoid cold starts
    scaledown_window=600,  # Keep container alive for 10 minutes
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI app for HTTP/WebSocket endpoints."""
    import uuid
    import asyncio
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    from starlette.middleware.base import BaseHTTPMiddleware
    
    web_app = FastAPI(title="LingBot-World API")
    
    # Custom CORS middleware to ensure headers are always present
    @web_app.middleware("http")
    async def add_cors_headers(request: Request, call_next):
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            return JSONResponse(
                content={},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "86400",
                }
            )
        
        # Process request and add CORS headers to response
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    
    # Also add standard CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Set to False when using "*" origins
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


# Function to check volume status
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=60,
)
def check_volume_status():
    """Check what's in the Modal volume."""
    print("=" * 60)
    print("Modal Volume Status Check")
    print("=" * 60)
    
    model_path = Path(MODEL_DIR) / "lingbot-world-base-cam"
    
    print(f"\nVolume mount: {MODEL_DIR}")
    print(f"Model path: {model_path}")
    print(f"Model path exists: {model_path.exists()}")
    
    if not Path(MODEL_DIR).exists():
        print("\n❌ Volume mount does not exist!")
        return
    
    # List all files
    all_files = list(Path(MODEL_DIR).rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    
    print(f"\nTotal items: {len(all_files)}")
    print(f"Total files: {len(files_only)}")
    
    if files_only:
        total_size = sum(f.stat().st_size for f in files_only)
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        
        # Show largest files
        print("\nLargest files:")
        for f in sorted(files_only, key=lambda x: x.stat().st_size, reverse=True)[:15]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {size_mb:>10.1f} MB  {f.relative_to(MODEL_DIR)}")
        
        # Check for safetensors
        safetensors = [f for f in files_only if f.suffix == ".safetensors"]
        print(f"\nSafetensor files: {len(safetensors)}")
        if safetensors:
            st_size = sum(f.stat().st_size for f in safetensors)
            print(f"Safetensor total size: {st_size / (1024**3):.2f} GB")
    else:
        print("\n❌ No files found in volume!")
    
    print("=" * 60)


# CLI command to download weights
@app.local_entrypoint()
def download_weights(force: bool = False, check_only: bool = False):
    """Download model weights to Modal volume.
    
    Args:
        force: Force re-download even if files exist
        check_only: Only check volume status, don't download
    """
    if check_only:
        print("Checking volume status...")
        check_volume_status.remote()
    else:
        print("Starting model download...")
        result = download_model_weights.remote(force=force)
        print(f"\n✓ Complete! Model at: {result}")
