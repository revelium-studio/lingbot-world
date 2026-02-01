"""
Control Adapter: Translates frontend input commands into LingBot-World camera poses.

LingBot-World expects:
- intrinsics.npy: Shape [num_frames, 4] with [fx, fy, cx, cy]
- poses.npy: Shape [num_frames, 4, 4] transformation matrices in OpenCV coordinates

This module converts WASD/mouse controls into these camera representations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CameraState:
    """Current camera state in world coordinates."""
    
    # Position in world space
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    # Euler angles in radians (yaw, pitch, roll)
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    # Camera intrinsics (fx, fy, cx, cy) - typical values for 480p
    intrinsics: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0, 416.0, 240.0]))
    
    def copy(self) -> "CameraState":
        """Create a deep copy of the camera state."""
        return CameraState(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            intrinsics=self.intrinsics.copy()
        )


# Control action types
ControlAction = Literal[
    "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right", "look_up", "look_down",
    "move_up", "move_down"
]


class ControlAdapter:
    """
    Adapts user input controls to LingBot-World camera pose format.
    
    Maintains camera state and generates pose sequences for the model.
    """
    
    # Movement speeds (units per frame)
    MOVE_SPEED = 0.1
    TURN_SPEED = 0.05  # radians per frame
    LOOK_SPEED = 0.03  # radians per frame
    
    def __init__(self, initial_state: CameraState | None = None):
        """Initialize with optional starting camera state."""
        self.state = initial_state or CameraState()
        self._pose_history: list[np.ndarray] = []
    
    def reset(self, resolution: tuple[int, int] = (480, 832)):
        """Reset camera to initial state with correct intrinsics for resolution."""
        height, width = resolution
        # Standard camera intrinsics based on resolution
        fx = fy = width * 0.6  # Approximate focal length
        cx, cy = width / 2, height / 2
        
        self.state = CameraState(
            intrinsics=np.array([fx, fy, cx, cy])
        )
        self._pose_history = []
    
    def apply_action(self, action: ControlAction) -> np.ndarray:
        """
        Apply a control action and return the new camera pose matrix.
        
        Args:
            action: The control action to apply
            
        Returns:
            4x4 transformation matrix in OpenCV coordinates
        """
        yaw, pitch, roll = self.state.rotation
        
        # Calculate forward and right vectors based on current rotation
        forward = np.array([
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch),
            np.cos(yaw) * np.cos(pitch)
        ])
        right = np.array([np.cos(yaw), 0, -np.sin(yaw)])
        up = np.array([0, 1, 0])
        
        # Apply movement based on action
        if action == "move_forward":
            self.state.position += forward * self.MOVE_SPEED
        elif action == "move_backward":
            self.state.position -= forward * self.MOVE_SPEED
        elif action == "move_left":
            self.state.position -= right * self.MOVE_SPEED
        elif action == "move_right":
            self.state.position += right * self.MOVE_SPEED
        elif action == "move_up":
            self.state.position += up * self.MOVE_SPEED
        elif action == "move_down":
            self.state.position -= up * self.MOVE_SPEED
        elif action == "turn_left":
            self.state.rotation[0] -= self.TURN_SPEED
        elif action == "turn_right":
            self.state.rotation[0] += self.TURN_SPEED
        elif action == "look_up":
            self.state.rotation[1] = np.clip(
                self.state.rotation[1] - self.LOOK_SPEED,
                -np.pi / 2 + 0.1,
                np.pi / 2 - 0.1
            )
        elif action == "look_down":
            self.state.rotation[1] = np.clip(
                self.state.rotation[1] + self.LOOK_SPEED,
                -np.pi / 2 + 0.1,
                np.pi / 2 - 0.1
            )
        
        pose = self._compute_pose_matrix()
        self._pose_history.append(pose)
        return pose
    
    def apply_mouse_delta(self, dx: float, dy: float, sensitivity: float = 0.002):
        """
        Apply mouse movement to camera rotation.
        
        Args:
            dx: Mouse delta X (pixels)
            dy: Mouse delta Y (pixels)
            sensitivity: Mouse sensitivity multiplier
        """
        self.state.rotation[0] += dx * sensitivity  # Yaw
        self.state.rotation[1] = np.clip(
            self.state.rotation[1] + dy * sensitivity,
            -np.pi / 2 + 0.1,
            np.pi / 2 - 0.1
        )
    
    def _compute_pose_matrix(self) -> np.ndarray:
        """
        Compute 4x4 transformation matrix from current state.
        
        Returns:
            4x4 matrix in OpenCV coordinate convention (right-handed, Y-down)
        """
        yaw, pitch, roll = self.state.rotation
        
        # Rotation matrices for each axis
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation (yaw * pitch * roll order)
        R = Ry @ Rx @ Rz
        
        # Build 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = self.state.position
        
        return pose
    
    def get_pose_sequence(self, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a pose sequence for the model.
        
        If we don't have enough history, interpolate/extrapolate.
        
        Args:
            num_frames: Number of frames to generate
            
        Returns:
            Tuple of (intrinsics, poses) arrays ready for the model
        """
        # Intrinsics are constant across frames
        intrinsics = np.tile(self.state.intrinsics, (num_frames, 1))
        
        if len(self._pose_history) == 0:
            # No history - use current pose for all frames
            current_pose = self._compute_pose_matrix()
            poses = np.tile(current_pose, (num_frames, 1, 1))
        elif len(self._pose_history) >= num_frames:
            # Enough history - use recent poses
            poses = np.array(self._pose_history[-num_frames:])
        else:
            # Partial history - pad with current pose
            current_pose = self._compute_pose_matrix()
            padding = num_frames - len(self._pose_history)
            poses = np.concatenate([
                np.tile(current_pose, (padding, 1, 1)),
                np.array(self._pose_history)
            ])
        
        return intrinsics, poses
    
    def get_current_pose(self) -> np.ndarray:
        """Get the current camera pose matrix."""
        return self._compute_pose_matrix()
    
    def get_current_intrinsics(self) -> np.ndarray:
        """Get the current camera intrinsics."""
        return self.state.intrinsics.copy()
