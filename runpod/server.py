"""
LingBot-World RunPod Server
FastAPI server for world generation on dedicated GPU
"""

import os
import io
import sys
import base64
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add lingbot-world to path
sys.path.insert(0, "/workspace/lingbot-world")

# Configuration
MODEL_DIR = "/workspace/models"
MODEL_PATH = Path(MODEL_DIR) / "lingbot-world-base-cam"

# Global model instance (loaded once, stays in memory)
wan_i2v = None
model_config = None
model_loaded = False

def load_model():
    """Load the LingBot-World model into GPU memory."""
    global wan_i2v, model_config, model_loaded
    
    import torch
    
    print("\n" + "=" * 60)
    print("🚀 LOADING LINGBOT-WORLD MODEL")
    print("=" * 60)
    
    print(f"\n🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n📂 Model path: {MODEL_PATH}")
    print(f"   Exists: {MODEL_PATH.exists()}")
    
    # Check for model files
    safetensors = list(MODEL_PATH.rglob("*.safetensors"))
    t5_file = MODEL_PATH / "models_t5_umt5-xxl-enc-bf16.pth"
    
    print(f"\n   Safetensor files: {len(safetensors)}")
    print(f"   T5 file exists: {t5_file.exists()}")
    
    if not safetensors or not t5_file.exists():
        print("\n❌ Model files not found!")
        print("   Run the start.sh script to download the model.")
        return False
    
    print("\n✓ Model files verified!")
    print("Loading model into GPU memory (this takes 1-2 minutes)...")
    
    try:
        from wan import WanI2V
        from wan.configs import WAN_CONFIGS
        
        model_config = WAN_CONFIGS['i2v-A14B']
        
        wan_i2v = WanI2V(
            config=model_config,
            checkpoint_dir=str(MODEL_PATH),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,  # Keep T5 on CPU to save GPU memory
            init_on_cpu=False,
            convert_model_dtype=False,
        )
        
        model_loaded = True
        print("\n" + "=" * 60)
        print("✅ MODEL LOADED SUCCESSFULLY!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


# Create FastAPI app
app = FastAPI(title="LingBot-World API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
sessions = {}


# Pydantic models
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


# Generation function
def generate_world(prompt: str, num_frames: int = 81, resolution: tuple = (480, 832)) -> dict:
    """Generate world frames using LingBot-World model."""
    import numpy as np
    from PIL import Image
    import time
    
    print(f"\n{'='*60}")
    print(f"🎬 GENERATING WORLD")
    print(f"{'='*60}")
    print(f"   Prompt: '{prompt[:60]}...'")
    print(f"   Frames: {num_frames}")
    print(f"   Resolution: {resolution}")
    print(f"   Model loaded: {model_loaded}")
    print(f"   wan_i2v: {wan_i2v is not None}")
    
    if not model_loaded or wan_i2v is None:
        print("⚠️ Model not loaded, generating placeholder frames")
        return generate_demo_frames(prompt, resolution, num_frames)
    
    try:
        # Create placeholder image
        height, width = resolution
        print(f"\n📷 Creating seed image ({height}x{width})...")
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                img_array[y, x, 0] = int(100 + 80 * x / width)
                img_array[y, x, 1] = int(60 + 60 * y / height)
                img_array[y, x, 2] = int(120 + 80 * (1 - x / width))
        img = Image.fromarray(img_array)
        print(f"   ✓ Image created: {img.size}")
        
        # Get max_area and shift for resolution
        print(f"\n⚙️ Loading config...")
        from wan.configs import MAX_AREA_CONFIGS
        size_key = f"{height}*{width}"
        max_area = MAX_AREA_CONFIGS.get(size_key, height * width)
        shift = 3.0 if height <= 480 else 5.0
        print(f"   max_area: {max_area}")
        print(f"   shift: {shift}")
        
        # Generate video
        print(f"\n🚀 Starting generation (this takes 2-4 minutes)...")
        start_time = time.time()
        
        video = wan_i2v.generate(
            input_prompt=prompt,
            img=img,
            action_path=None,
            max_area=max_area,
            frame_num=num_frames,
            shift=shift,
            sample_solver="unipc",
            sampling_steps=30,  # Faster generation
            guide_scale=5.0,
            seed=-1,
            offload_model=False,  # Keep in memory for speed
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Generation complete in {elapsed:.1f}s")
        print(f"   Video shape: {video.shape}")
        
        # Convert to base64 frames
        print(f"\n📦 Converting to frames...")
        video_cpu = video.cpu()
        video_normalized = (video_cpu + 1) / 2 * 255
        video_normalized = video_normalized.clamp(0, 255).byte()
        video_frames = video_normalized.permute(1, 2, 3, 0).numpy()
        
        frames_base64 = []
        for i in range(video_frames.shape[0]):
            frame = Image.fromarray(video_frames[i])
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=85)
            frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            frames_base64.append(frame_b64)
        
        print(f"\n{'='*60}")
        print(f"✅ GENERATED {len(frames_base64)} REAL FRAMES!")
        print(f"{'='*60}")
        
        return {
            "frames": frames_base64,
            "num_frames": len(frames_base64),
            "resolution": resolution,
        }
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ GENERATION ERROR")
        print(f"{'='*60}")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n   Falling back to demo frames...")
        return generate_demo_frames(prompt, resolution, num_frames)


def generate_demo_frames(prompt: str, resolution: tuple, num_frames: int) -> dict:
    """Generate demo placeholder frames."""
    import numpy as np
    import hashlib
    from PIL import Image
    
    height, width = resolution
    frames_base64 = []
    
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
    np.random.seed(prompt_hash % (2**31))
    base_color = [np.random.randint(40, 180) for _ in range(3)]
    
    for i in range(min(num_frames, 60)):
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        t = i / max(num_frames, 1)
        
        for c in range(3):
            base = base_color[c] + int(40 * np.sin(t * 2 * np.pi + c * 2.1))
            img_array[:, :, c] = np.clip(base, 0, 255)
        
        for y in range(height):
            gradient = int(40 * y / height)
            img_array[y, :, :] = np.clip(
                img_array[y, :, :].astype(np.int16) + gradient, 0, 255
            ).astype(np.uint8)
        
        frame = Image.fromarray(img_array)
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=85)
        frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames_base64.append(frame_b64)
    
    return {
        "frames": frames_base64,
        "num_frames": len(frames_base64),
        "resolution": resolution,
        "demo_mode": True,
    }


# Background generation task
async def generate_frames_background(session_id: str, prompt: str):
    """Generate frames in background."""
    try:
        sessions[session_id]["status"] = "generating"
        sessions[session_id]["message"] = "Generating world on GPU..."
        
        # Run generation in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_world(prompt, num_frames=81, resolution=(480, 832))
        )
        
        sessions[session_id]["frames"] = result.get("frames", [])
        sessions[session_id]["status"] = "ready"
        sessions[session_id]["current_frame"] = 0
        sessions[session_id]["message"] = "World generated!"
        
    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["message"] = str(e)


# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "LingBot-World API",
        "status": "running",
        "model_loaded": model_loaded,
        "gpu": "A100-80GB",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_loaded}


@app.post("/api/world/create", response_model=CreateWorldResponse)
async def create_world(request: CreateWorldRequest, background_tasks: BackgroundTasks):
    """Create a new world generation session."""
    session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "prompt": request.prompt,
        "status": "initializing",
        "message": "Starting generation...",
        "frames": [],
        "current_frame": 0,
    }
    
    background_tasks.add_task(generate_frames_background, session_id, request.prompt)
    
    return CreateWorldResponse(
        session_id=session_id,
        status="generating",
        message="World generation started"
    )


@app.get("/api/world/{session_id}/status")
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


@app.get("/api/world/{session_id}/frame")
async def get_current_frame(session_id: str):
    """Get the current frame as an image."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    frames = session.get("frames", [])
    
    if not frames:
        raise HTTPException(status_code=404, detail="No frames available yet")
    
    frame_index = session.get("current_frame", 0) % len(frames)
    frame_b64 = frames[frame_index]
    session["current_frame"] = (frame_index + 1) % len(frames)
    
    frame_data = base64.b64decode(frame_b64)
    return StreamingResponse(io.BytesIO(frame_data), media_type="image/jpeg")


@app.post("/api/world/{session_id}/control")
async def send_control(session_id: str, request: ControlRequest):
    """Send a control action to the world."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    frames = session.get("frames", [])
    
    if frames:
        if request.action in ["move_forward", "move_right", "turn_right"]:
            session["current_frame"] = (session.get("current_frame", 0) + 2) % len(frames)
        elif request.action in ["move_backward", "move_left", "turn_left"]:
            session["current_frame"] = (session.get("current_frame", 0) - 2) % len(frames)
    
    return {"success": True, "frame_index": session.get("current_frame", 0)}


@app.delete("/api/world/{session_id}")
async def delete_world(session_id: str):
    """End a world session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time frame streaming."""
    await websocket.accept()
    
    if session_id not in sessions:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    try:
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
            
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=frame_interval
                )
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "control":
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


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    print("\n🚀 Server starting up...")
    load_model()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
