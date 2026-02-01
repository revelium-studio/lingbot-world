"""
World Generator: Wrapper around LingBot-World model for interactive world generation.

This module handles:
- Model loading and initialization using wan.WanI2V
- Image-to-video generation with text prompts
- Streaming frame generation
- Integration with control signals (camera poses in OpenCV format)

Based on: https://github.com/Robbyant/lingbot-world
HuggingFace: https://huggingface.co/robbyant/lingbot-world-base-cam
"""

import gc
import os
import sys
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for world generation."""
    prompt: str
    initial_image: Image.Image | None = None
    resolution: tuple[int, int] = (480, 832)  # height, width
    num_frames: int = 161
    fps: int = 16
    # Diffusion sampling parameters
    sampling_steps: int = 40
    guide_scale: float = 5.0
    shift: float = 3.0  # 3.0 for 480p, 5.0 for 720p
    sample_solver: str = "unipc"  # "unipc" or "dpm++"
    seed: int = -1  # -1 for random


@dataclass
class GeneratedFrame:
    """A single generated frame with metadata."""
    frame_index: int
    image: Image.Image
    timestamp_ms: float


class WorldGenerator:
    """
    Manages LingBot-World model loading and generation.
    
    This class wraps the LingBot-World inference pipeline (wan.WanI2V) to provide:
    - Async-friendly generation
    - Streaming frame output
    - Control signal integration via camera poses
    
    The model uses:
    - Task: i2v-A14B (Image-to-Video with 14B parameter model)
    - Control: Camera poses in OpenCV coordinate format
    - Output: Video frames at 16 FPS
    """
    
    def __init__(self):
        self._wan_i2v = None
        self._config = None
        self._is_loaded = False
        self._device_id = 0
        self._lock = asyncio.Lock()
        self._temp_action_dir = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._is_loaded
    
    async def load_model(self) -> None:
        """
        Load the LingBot-World model into memory.
        
        This should be called once at startup. The model stays loaded
        for the lifetime of the application.
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return
        
        async with self._lock:
            if self._is_loaded:
                return
            
            logger.info("Loading LingBot-World model...")
            
            # Add LingBot-World repo to path if needed
            repo_path = settings.lingbot_repo_path
            if repo_path.exists() and str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
                logger.info(f"Added {repo_path} to Python path")
            
            # Determine device
            if settings.effective_device == "cuda":
                self._device_id = 0
            else:
                self._device_id = "cpu"
            
            logger.info(f"Using device: {settings.effective_device}")
            
            # Load model components in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_load_components)
            
            if self._wan_i2v is not None:
                self._is_loaded = True
                logger.info("LingBot-World model loaded successfully")
            else:
                logger.warning("Model loading failed, running in demo mode")
    
    def _sync_load_components(self) -> None:
        """
        Synchronous model component loading.
        
        Loads the WanI2V pipeline with the i2v-A14B configuration.
        """
        model_path = settings.model_path
        
        if not model_path.exists():
            logger.warning(
                f"Model path {model_path} does not exist. "
                "Please run: python scripts/download_weights.py"
            )
            self._wan_i2v = None
            return
        
        try:
            # Import LingBot-World modules
            import wan
            from wan.configs import WAN_CONFIGS
            
            # Get configuration for i2v-A14B task
            self._config = WAN_CONFIGS['i2v-A14B']
            
            logger.info(f"Creating WanI2V pipeline from {model_path}")
            
            # Initialize the WanI2V model
            # Note: For single GPU without distributed training,
            # we disable FSDP and sequence parallel
            self._wan_i2v = wan.WanI2V(
                config=self._config,
                checkpoint_dir=str(model_path),
                device_id=self._device_id if isinstance(self._device_id, int) else 0,
                rank=0,
                t5_fsdp=False,  # Disable for single GPU
                dit_fsdp=False,  # Disable for single GPU
                use_sp=False,   # Disable sequence parallel
                t5_cpu=settings.use_t5_cpu,
                init_on_cpu=True,
                convert_model_dtype=False,
            )
            
            logger.info("WanI2V pipeline created successfully")
            
        except ImportError as e:
            logger.error(
                f"Could not import LingBot-World modules: {e}. "
                f"Make sure the repo is cloned to {settings.lingbot_repo_path} "
                "and all dependencies are installed."
            )
            self._wan_i2v = None
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            self._wan_i2v = None
    
    async def generate_world(
        self,
        config: GenerationConfig,
        intrinsics: np.ndarray | None = None,
        poses: np.ndarray | None = None,
        on_frame: Callable[[GeneratedFrame], None] | None = None
    ) -> AsyncIterator[GeneratedFrame]:
        """
        Generate a world from prompt and optional initial image.
        
        Args:
            config: Generation configuration
            intrinsics: Optional camera intrinsics [num_frames, 4] as [fx, fy, cx, cy]
            poses: Optional camera poses [num_frames, 4, 4] in OpenCV coordinates
            on_frame: Optional callback for each generated frame
            
        Yields:
            GeneratedFrame objects as they are generated
        """
        if not self._is_loaded:
            await self.load_model()
        
        logger.info(f"Generating world with prompt: {config.prompt[:50]}...")
        
        # If no initial image provided, create a placeholder
        if config.initial_image is None:
            config.initial_image = self._create_placeholder_image(config.resolution)
        
        # Generate frames
        frame_index = 0
        start_time = asyncio.get_event_loop().time()
        
        async for frame_data in self._generate_frames(config, intrinsics, poses):
            current_time = asyncio.get_event_loop().time()
            frame = GeneratedFrame(
                frame_index=frame_index,
                image=frame_data,
                timestamp_ms=(current_time - start_time) * 1000
            )
            
            if on_frame:
                on_frame(frame)
            
            yield frame
            frame_index += 1
    
    async def _generate_frames(
        self,
        config: GenerationConfig,
        intrinsics: np.ndarray | None,
        poses: np.ndarray | None
    ) -> AsyncIterator[Image.Image]:
        """
        Core frame generation loop.
        
        Calls the actual LingBot-World inference via WanI2V.generate().
        """
        if self._wan_i2v is None:
            # Demo mode: generate placeholder frames
            logger.warning("Running in demo mode (no model loaded)")
            async for frame in self._generate_demo_frames(config):
                yield frame
            return
        
        # Prepare action path if poses are provided
        action_path = None
        if poses is not None and intrinsics is not None:
            action_path = self._prepare_action_path(poses, intrinsics)
        
        # Run inference in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        video_tensor = await loop.run_in_executor(
            None,
            self._sync_generate,
            config,
            action_path
        )
        
        if video_tensor is None:
            logger.error("Generation failed, falling back to demo mode")
            async for frame in self._generate_demo_frames(config):
                yield frame
            return
        
        # Convert video tensor to PIL images and yield
        # Video tensor shape: (C, N, H, W) where C=3, N=num_frames
        frames = self._tensor_to_frames(video_tensor)
        
        # Yield frames with appropriate timing for ~16 FPS
        frame_interval = 1.0 / config.fps
        
        for frame in frames:
            yield frame
            await asyncio.sleep(frame_interval)
    
    def _prepare_action_path(
        self,
        poses: np.ndarray,
        intrinsics: np.ndarray
    ) -> str:
        """
        Prepare a temporary directory with camera control files.
        
        LingBot-World expects action_path to contain:
        - poses.npy: [num_frames, 4, 4] transformation matrices (OpenCV coords)
        - intrinsics.npy: [num_frames, 4] with [fx, fy, cx, cy]
        """
        if self._temp_action_dir is None:
            self._temp_action_dir = tempfile.mkdtemp(prefix="lingbot_action_")
        
        # Save poses and intrinsics
        poses_path = os.path.join(self._temp_action_dir, "poses.npy")
        intrinsics_path = os.path.join(self._temp_action_dir, "intrinsics.npy")
        
        np.save(poses_path, poses.astype(np.float32))
        np.save(intrinsics_path, intrinsics.astype(np.float32))
        
        logger.debug(f"Saved action files to {self._temp_action_dir}")
        return self._temp_action_dir
    
    def _sync_generate(
        self,
        config: GenerationConfig,
        action_path: str | None
    ) -> torch.Tensor | None:
        """
        Synchronous generation using the WanI2V model.
        
        Returns video tensor of shape (C, N, H, W) or None on failure.
        """
        try:
            from wan.configs import MAX_AREA_CONFIGS
            
            # Determine max_area based on resolution
            height, width = config.resolution
            size_key = f"{height}*{width}"
            max_area = MAX_AREA_CONFIGS.get(size_key, height * width)
            
            # Adjust shift for resolution (3.0 for 480p, 5.0 for 720p)
            shift = 3.0 if height <= 480 else 5.0
            
            logger.info(
                f"Generating: resolution={size_key}, frames={config.num_frames}, "
                f"steps={config.sampling_steps}, action_path={action_path is not None}"
            )
            
            # Call WanI2V.generate()
            video = self._wan_i2v.generate(
                input_prompt=config.prompt,
                img=config.initial_image,
                action_path=action_path,
                max_area=max_area,
                frame_num=config.num_frames,
                shift=shift,
                sample_solver=config.sample_solver,
                sampling_steps=config.sampling_steps,
                guide_scale=config.guide_scale,
                seed=config.seed,
                offload_model=True,  # Save VRAM by offloading between steps
            )
            
            return video
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return None
    
    def _tensor_to_frames(self, video_tensor: torch.Tensor) -> list[Image.Image]:
        """
        Convert video tensor to list of PIL Images.
        
        Args:
            video_tensor: Shape (C, N, H, W), values in [-1, 1]
            
        Returns:
            List of PIL Images
        """
        # Normalize from [-1, 1] to [0, 255]
        video = video_tensor.cpu()
        video = (video + 1) / 2 * 255
        video = video.clamp(0, 255).byte()
        
        # video shape: (C, N, H, W) -> need (N, H, W, C)
        video = video.permute(1, 2, 3, 0).numpy()
        
        frames = []
        for i in range(video.shape[0]):
            frame = Image.fromarray(video[i])
            frames.append(frame)
        
        return frames
    
    async def _generate_demo_frames(
        self,
        config: GenerationConfig
    ) -> AsyncIterator[Image.Image]:
        """Generate demo frames when model is not available."""
        import random
        
        height, width = config.resolution
        frame_interval = 1.0 / config.fps
        
        # Generate a series of evolving demo frames with prompt-based color
        # Use hash of prompt to get consistent colors for same prompt
        prompt_hash = hash(config.prompt) % 1000000
        random.seed(prompt_hash)
        base_color = [random.randint(30, 180) for _ in range(3)]
        random.seed()  # Reset seed
        
        for i in range(min(config.num_frames, 60)):  # Cap at 60 for demo
            # Create animated gradient with prompt-based colors
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Animate colors based on frame number
            t = i / max(config.num_frames, 1)
            for c in range(3):
                value = int(base_color[c] + 50 * np.sin(t * 2 * np.pi + c * 2.1))
                img_array[:, :, c] = np.clip(value, 0, 255)
            
            # Add gradient and noise for visual interest
            for y in range(height):
                for c in range(3):
                    gradient = int(30 * y / height)
                    img_array[y, :, c] = np.clip(
                        img_array[y, :, c].astype(np.int16) + gradient, 0, 255
                    ).astype(np.uint8)
            
            # Add some noise
            noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
            img_array = np.clip(
                img_array.astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)
            
            yield Image.fromarray(img_array)
            await asyncio.sleep(frame_interval)
    
    def _create_placeholder_image(self, resolution: tuple[int, int]) -> Image.Image:
        """Create a placeholder initial image with gradient."""
        height, width = resolution
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a nice gradient
        for y in range(height):
            for x in range(width):
                img_array[y, x, 0] = int(100 + 80 * x / width)      # Red
                img_array[y, x, 1] = int(60 + 60 * y / height)      # Green  
                img_array[y, x, 2] = int(120 + 80 * (1 - x / width)) # Blue
        
        return Image.fromarray(img_array)
    
    async def update_with_action(
        self,
        session_prompt: str,
        session_image: Image.Image,
        action_poses: np.ndarray,
        action_intrinsics: np.ndarray
    ) -> AsyncIterator[GeneratedFrame]:
        """
        Update world generation with new control actions.
        
        This method generates new frames conditioned on the camera poses,
        allowing real-time navigation in the generated world.
        
        Args:
            session_prompt: The original world prompt
            session_image: Current view image to continue from
            action_poses: Camera poses [num_frames, 4, 4]
            action_intrinsics: Camera intrinsics [num_frames, 4]
            
        Yields:
            New generated frames
        """
        logger.info("Updating world with new camera actions")
        
        # Create config for continuation generation
        config = GenerationConfig(
            prompt=session_prompt,
            initial_image=session_image,
            resolution=settings.resolution,
            num_frames=min(17, len(action_poses)),  # Short sequences for responsiveness
            fps=settings.default_fps,
            sampling_steps=20,  # Fewer steps for faster response
        )
        
        frame_index = 0
        async for frame in self._generate_frames(config, action_intrinsics, action_poses):
            yield GeneratedFrame(
                frame_index=frame_index,
                image=frame,
                timestamp_ms=0
            )
            frame_index += 1
    
    def unload_model(self) -> None:
        """Unload the model from memory and clean up resources."""
        if self._wan_i2v is not None:
            # Clean up model components
            del self._wan_i2v
            self._wan_i2v = None
        
        self._config = None
        self._is_loaded = False
        
        # Clean up temporary action directory
        if self._temp_action_dir and os.path.exists(self._temp_action_dir):
            import shutil
            try:
                shutil.rmtree(self._temp_action_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir: {e}")
            self._temp_action_dir = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded and resources cleaned up")


# Singleton instance for the application
world_generator = WorldGenerator()
