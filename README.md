# LingBot-World Interactive Application

An interactive web application for exploring AI-generated worlds using the open-source **LingBot-World** model.

[![GitHub](https://img.shields.io/badge/Model-Robbyant%2Flingbot--world-blue)](https://github.com/Robbyant/lingbot-world)
[![HuggingFace](https://img.shields.io/badge/🤗-lingbot--world--base--cam-yellow)](https://huggingface.co/robbyant/lingbot-world-base-cam)
[![arXiv](https://img.shields.io/badge/arXiv-2601.20540-b31b1b)](https://arxiv.org/abs/2601.20540)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

## ✨ Features

- **Text-to-World Generation**: Describe any world in natural language and watch it come to life
- **Real-time Exploration**: Navigate the generated world with WASD/mouse controls
- **Live Streaming**: Frames are streamed at ~16 FPS via WebSocket
- **Camera Control**: Your movement controls are translated to camera poses for the model
- **High Fidelity**: Supports 480P and 720P resolution output

## 🎬 About LingBot-World

LingBot-World is an open-source world simulator from the Robbyant Team, built on the Wan2.2 architecture. Key capabilities:

- **High-Fidelity Environments**: Realistic, scientific, cartoon, and diverse visual styles
- **Long-Term Memory**: Minute-level video generation with temporal consistency
- **Real-Time Interactivity**: <1 second latency at 16 FPS throughput
- **Open Access**: Apache 2.0 licensed model and code

## 🏗️ Architecture

```
lingbot-world/
├── backend/                     # FastAPI Python backend
│   ├── api/                     # REST & WebSocket endpoints
│   ├── model/
│   │   ├── world_generator.py   # WanI2V wrapper for inference
│   │   └── control_adapter.py   # WASD → OpenCV camera poses
│   └── session/                 # Multi-user session management
├── frontend/                    # React + Vite frontend
│   └── src/
│       ├── components/          # Prompt, Loading, WorldView screens
│       └── styles/              # Dark cosmic theme
├── lingbot-world-repo/          # Cloned LingBot-World repository
├── lingbot-world-base-cam/      # Downloaded model weights
└── scripts/                     # Setup & development helpers
```

## 📋 Requirements

| Component | Requirement |
|-----------|------------|
| Python | 3.10+ |
| Node.js | 18+ |
| PyTorch | ≥2.4.0 |
| GPU VRAM | 16GB+ (recommended) |
| CUDA | 11.8+ (for GPU) |

### Key Dependencies

From the [official requirements](https://github.com/Robbyant/lingbot-world/blob/main/requirements.txt):

- `torch>=2.4.0`, `torchvision>=0.19.0`, `torchaudio`
- `diffusers>=0.31.0`
- `transformers>=4.49.0,<=4.51.3`
- `accelerate>=1.1.1`
- `flash-attn` (GPU acceleration, installed separately)
- `opencv-python>=4.9.0.80`
- `numpy>=1.23.5,<2`

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd lingbot-world

# Run the setup script (installs everything including model weights)
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The setup script will:
1. Create Python virtual environment
2. Install PyTorch with CUDA support (if available)
3. Install all Python dependencies
4. Clone [Robbyant/lingbot-world](https://github.com/Robbyant/lingbot-world) repository
5. Download model weights from [HuggingFace](https://huggingface.co/robbyant/lingbot-world-base-cam) (~15GB)
6. Install `flash-attn` for GPU acceleration
7. Install frontend npm packages

### 2. Start Development Servers

**Option A: Single command**
```bash
./scripts/start_dev.sh
```

**Option B: Manual startup**

Terminal 1 (Backend):
```bash
source venv/bin/activate
python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

### 3. Open the App

Navigate to **http://localhost:5173** in your browser.

## 🎮 Controls

| Control | Action |
|---------|--------|
| **W/A/S/D** | Move forward/left/backward/right |
| **Q/E** | Turn left/right |
| **Arrow Keys** | Look up/down/left/right |
| **Space** | Move up |
| **Shift** | Move down |
| **Mouse** | Look around (click viewport to enable) |
| **Esc** | Release mouse |

## ⚙️ Configuration

Edit `.env` in the project root:

```env
# Model settings
MODEL_PATH=./lingbot-world-base-cam
MODEL_SIZE=480*832    # Options: 480*832, 720*1280
FRAME_NUM=161         # Must be 4n+1

# Device (auto, cuda, cpu)
DEVICE=auto

# Memory optimization
USE_T5_CPU=false      # Set true to save ~2GB VRAM

# Server ports
BACKEND_PORT=8000
FRONTEND_PORT=5173
```

### Resolution Options

| Size | Resolution | Shift | VRAM Required |
|------|-----------|-------|---------------|
| `480*832` | 480P | 3.0 | ~14GB |
| `720*1280` | 720P | 5.0 | ~20GB |

### Model Variants

| Model | Control Signals | Status |
|-------|-----------------|--------|
| [lingbot-world-base-cam](https://huggingface.co/robbyant/lingbot-world-base-cam) | Camera Poses | ✅ Available |
| lingbot-world-base-act | Actions | 🔜 Coming soon |
| lingbot-world-fast | - | 🔜 Coming soon |

## 📡 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/status` | GET | Server & model status |
| `POST /api/world/create` | POST | Create new world session |
| `GET /api/world/{id}/status` | GET | Session status |
| `GET /api/world/{id}/frame` | GET | Get current frame (JPEG) |
| `POST /api/world/{id}/control` | POST | Send control action |
| `DELETE /api/world/{id}` | DELETE | End session |

### WebSocket

Connect to `/ws/{session_id}` for real-time streaming:

**Server → Client (frames):**
```json
{
  "type": "frame",
  "frame_index": 42,
  "data": "<base64 JPEG>",
  "is_generating": true
}
```

**Client → Server (controls):**
```json
{
  "type": "control",
  "action": "move_forward",
  "mouse_dx": 0,
  "mouse_dy": 0
}
```

## 🧠 How It Works

1. **Prompt Processing**: User enters a world description
2. **Model Loading**: Backend loads WanI2V pipeline (first time only)
3. **Image-to-Video Generation**: Model generates frames from text + initial image
4. **Camera Conditioning**: User controls are converted to OpenCV pose matrices
5. **Frame Streaming**: Frames sent via WebSocket at ~16 FPS

### Camera Control Format

LingBot-World expects camera poses as:
- `poses.npy`: Shape `[num_frames, 4, 4]` - OpenCV transformation matrices
- `intrinsics.npy`: Shape `[num_frames, 4]` - `[fx, fy, cx, cy]`

The `ControlAdapter` handles this conversion from WASD/mouse input.

## 🐛 Troubleshooting

### Model not loading

```bash
# Check if weights are downloaded
ls -la lingbot-world-base-cam/

# Re-download if needed
python scripts/download_weights.py --model base-cam
```

### Out of GPU memory

```bash
# Edit .env
USE_T5_CPU=true  # Saves ~2GB VRAM
```

Or reduce resolution:
```bash
MODEL_SIZE=480*832
```

### flash-attn installation fails

```bash
# Ensure CUDA toolkit is installed
nvcc --version

# Install manually
pip install flash-attn --no-build-isolation
```

### Import errors for 'wan' module

```bash
# Ensure repo is cloned and installed
cd lingbot-world-repo
pip install -e .
```

## 📚 References

- **GitHub**: [Robbyant/lingbot-world](https://github.com/Robbyant/lingbot-world)
- **HuggingFace**: [robbyant/lingbot-world-base-cam](https://huggingface.co/robbyant/lingbot-world-base-cam)
- **Paper**: [arXiv:2601.20540](https://arxiv.org/abs/2601.20540)
- **Project Page**: [technology.robbyant.com/lingbot-world](https://technology.robbyant.com/lingbot-world)

## 📄 License

This project is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.

The LingBot-World model is Apache 2.0 licensed by the Robbyant Team.

## 🙏 Acknowledgments

- [Robbyant Team](https://github.com/Robbyant/lingbot-world) for the LingBot-World model
- [Wan2.2 Team](https://github.com/Wan-Video/Wan2.1) for the foundation architecture

## 📖 Citation

```bibtex
@article{lingbot-world,
  title={Advancing Open-source World Models}, 
  author={Robbyant Team},
  journal={arXiv preprint arXiv:2601.20540},
  year={2026}
}
```

---

**Demo Mode**: The app includes a demo mode with placeholder frames, allowing you to develop and test the UI without downloading the 15GB model weights.
