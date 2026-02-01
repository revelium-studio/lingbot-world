# Example Inputs

This directory contains example input files for LingBot-World generation.

## Structure

Each example should be in its own subdirectory:

```
examples/
├── 00/
│   ├── image.jpg        # Starting image
│   ├── intrinsics.npy   # Camera intrinsics [num_frames, 4]
│   └── poses.npy        # Camera poses [num_frames, 4, 4]
├── 01/
│   └── ...
```

## File Formats

### `intrinsics.npy`
- Shape: `[num_frames, 4]`
- Values: `[fx, fy, cx, cy]` (focal length and principal point)

### `poses.npy`
- Shape: `[num_frames, 4, 4]`
- 4x4 transformation matrices in OpenCV coordinates

## Generating Control Signals

Camera poses can be extracted from existing videos using ViPE (Visual Pose Estimator).

For manual creation, the `ControlAdapter` class in `backend/model/control_adapter.py` can generate these files from WASD/mouse inputs.
