"""
Configuration settings for the LingBot-World application.

Based on: https://github.com/Robbyant/lingbot-world
HuggingFace: https://huggingface.co/robbyant/lingbot-world-base-cam
"""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================
    # Model paths
    # ==========================================
    model_path: Path = Path("./lingbot-world-base-cam")
    lingbot_repo_path: Path = Path("./lingbot-world-repo")
    
    # Model configuration
    # Supported sizes: 480*832, 832*480, 720*1280, 1280*720
    model_size: str = "480*832"
    frame_num: int = 161  # Must be 4n+1 (e.g., 17, 81, 161, 321, 961)
    
    # ==========================================
    # Device settings
    # ==========================================
    device: Literal["auto", "cuda", "cpu"] = "auto"
    
    # ==========================================
    # Server settings
    # ==========================================
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_port: int = 5173
    
    # ==========================================
    # Generation settings
    # ==========================================
    default_fps: int = 16
    max_concurrent_sessions: int = 4
    
    # Diffusion sampling defaults
    default_sampling_steps: int = 40
    default_guide_scale: float = 5.0
    
    # ==========================================
    # Memory optimization
    # ==========================================
    # Place T5 text encoder on CPU to save ~2GB VRAM
    use_t5_cpu: bool = False
    
    # FSDP (Fully Sharded Data Parallel) for multi-GPU
    use_fsdp: bool = False
    
    # ==========================================
    # Paths
    # ==========================================
    examples_path: Path = Path("./examples")
    output_path: Path = Path("./outputs")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars
    
    @property
    def resolution(self) -> tuple[int, int]:
        """Parse model_size into (height, width) tuple."""
        parts = self.model_size.split("*")
        if len(parts) != 2:
            return (480, 832)  # Default fallback
        return int(parts[0]), int(parts[1])
    
    @property
    def effective_device(self) -> str:
        """Determine the actual device to use."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    @property
    def max_area(self) -> int:
        """Calculate max pixel area for the model."""
        height, width = self.resolution
        return height * width
    
    def get_shift_for_resolution(self) -> float:
        """
        Get the recommended shift parameter for the current resolution.
        
        From LingBot-World docs:
        - 480p: shift = 3.0
        - 720p: shift = 5.0
        """
        height, _ = self.resolution
        return 3.0 if height <= 480 else 5.0


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure required directories exist."""
    settings.output_path.mkdir(parents=True, exist_ok=True)
    settings.examples_path.mkdir(parents=True, exist_ok=True)


def validate_model_path() -> bool:
    """Check if model weights are downloaded."""
    model_path = settings.model_path
    if not model_path.exists():
        return False
    
    # Check for essential model files
    # LingBot-World uses safetensors format
    model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
    return len(model_files) > 0


def validate_repo_path() -> bool:
    """Check if LingBot-World repo is cloned."""
    repo_path = settings.lingbot_repo_path
    if not repo_path.exists():
        return False
    
    # Check for wan module
    wan_init = repo_path / "wan" / "__init__.py"
    return wan_init.exists()
