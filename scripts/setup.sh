#!/bin/bash
# LingBot-World Interactive Application Setup Script
# Based on: https://github.com/Robbyant/lingbot-world
# HuggingFace: https://huggingface.co/robbyant/lingbot-world-base-cam

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║       LingBot-World Interactive App Setup                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Check Python version (requires 3.10+)
echo -e "${CYAN}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10+ is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check Node.js
echo -e "${CYAN}Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed. Please install Node.js 18+${NC}"
    exit 1
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Node.js $NODE_VERSION${NC}"

# Check for CUDA
echo ""
echo -e "${CYAN}Checking CUDA availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓ NVIDIA GPU detected (Driver: $CUDA_VERSION)${NC}"
    HAS_CUDA=true
else
    echo -e "${YELLOW}! No NVIDIA GPU detected. Model will run on CPU (slower).${NC}"
    HAS_CUDA=false
fi

# Create virtual environment
echo ""
echo -e "${CYAN}Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Created virtual environment${NC}"
else
    echo -e "${YELLOW}! Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo -e "${CYAN}Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Install PyTorch first (for CUDA support)
echo ""
echo -e "${CYAN}Installing PyTorch...${NC}"
if [ "$HAS_CUDA" = true ]; then
    # Install PyTorch with CUDA support
    pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    # Install CPU-only PyTorch
    pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Install Python dependencies
echo ""
echo -e "${CYAN}Installing Python dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Install flash-attn (GPU only, optional but recommended)
if [ "$HAS_CUDA" = true ]; then
    echo ""
    echo -e "${CYAN}Installing flash-attn (GPU acceleration)...${NC}"
    echo "This may take several minutes as it compiles CUDA kernels..."
    if pip install flash-attn --no-build-isolation 2>/dev/null; then
        echo -e "${GREEN}✓ flash-attn installed${NC}"
    else
        echo -e "${YELLOW}! flash-attn installation failed.${NC}"
        echo -e "${YELLOW}  This is optional but provides faster inference.${NC}"
        echo -e "${YELLOW}  You can try manually: pip install flash-attn --no-build-isolation${NC}"
    fi
fi

# Clone LingBot-World repository
echo ""
echo -e "${CYAN}Setting up LingBot-World repository...${NC}"
if [ ! -d "lingbot-world-repo" ]; then
    echo "Cloning from https://github.com/Robbyant/lingbot-world.git..."
    git clone https://github.com/Robbyant/lingbot-world.git lingbot-world-repo
    echo -e "${GREEN}✓ LingBot-World repository cloned${NC}"
else
    echo -e "${YELLOW}! LingBot-World repository already exists, updating...${NC}"
    cd lingbot-world-repo
    git pull origin main 2>/dev/null || echo "Could not update (may be offline)"
    cd ..
fi

# Install LingBot-World as a package (for wan module imports)
echo ""
echo -e "${CYAN}Installing LingBot-World package...${NC}"
pip install -e lingbot-world-repo/
echo -e "${GREEN}✓ LingBot-World package installed${NC}"

# Download model weights
echo ""
echo -e "${CYAN}Checking model weights...${NC}"
if [ ! -d "lingbot-world-base-cam" ] || [ -z "$(ls -A lingbot-world-base-cam 2>/dev/null)" ]; then
    echo "Model weights not found. Downloading from HuggingFace..."
    echo -e "${YELLOW}This will download ~15GB of model weights.${NC}"
    echo ""
    
    # Install huggingface-cli if needed
    pip install "huggingface_hub[cli]" 2>/dev/null || true
    
    echo "Downloading robbyant/lingbot-world-base-cam..."
    huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam
    
    echo -e "${GREEN}✓ Model weights downloaded${NC}"
else
    echo -e "${GREEN}✓ Model weights already present${NC}"
fi

# Install frontend dependencies
echo ""
echo -e "${CYAN}Installing frontend dependencies...${NC}"
cd frontend
npm install
cd ..
echo -e "${GREEN}✓ Frontend dependencies installed${NC}"

# Create output directories
echo ""
echo -e "${CYAN}Creating directories...${NC}"
mkdir -p outputs examples
echo -e "${GREEN}✓ Directories created${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo -e "${CYAN}Creating .env configuration file...${NC}"
    cat > .env << 'EOF'
# LingBot-World Interactive App Configuration
# Based on: https://github.com/Robbyant/lingbot-world

# ============================================
# Model Settings
# ============================================
MODEL_PATH=./lingbot-world-base-cam
MODEL_SIZE=480*832
FRAME_NUM=161

# ============================================
# Device Settings
# ============================================
# Options: auto, cuda, cpu
# 'auto' will use CUDA if available, otherwise CPU
DEVICE=auto

# ============================================
# Server Settings
# ============================================
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_PORT=5173

# ============================================
# Generation Settings
# ============================================
DEFAULT_FPS=16
MAX_CONCURRENT_SESSIONS=4

# ============================================
# Memory Optimization
# ============================================
# Set USE_T5_CPU=true if running out of GPU memory
# This moves the T5 text encoder to CPU (~2GB VRAM savings)
USE_T5_CPU=false

# FSDP (Fully Sharded Data Parallel) - for multi-GPU setups
USE_FSDP=false

# ============================================
# LingBot-World Repository Path
# ============================================
LINGBOT_REPO_PATH=./lingbot-world-repo
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}To start the application:${NC}"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start the backend server:"
echo "     python -m backend.main"
echo ""
echo "  3. In a new terminal, start the frontend:"
echo "     cd frontend && npm run dev"
echo ""
echo "  4. Open http://localhost:5173 in your browser"
echo ""
echo -e "${YELLOW}Notes:${NC}"
echo "  • First generation may take longer as the model loads (~30s)"
echo "  • Requires ~16GB GPU VRAM for full model"
echo "  • Set USE_T5_CPU=true in .env if running low on VRAM"
echo "  • Without GPU, generation will be significantly slower"
echo ""
echo -e "${CYAN}Model info:${NC}"
echo "  • Repository: https://github.com/Robbyant/lingbot-world"
echo "  • HuggingFace: https://huggingface.co/robbyant/lingbot-world-base-cam"
echo "  • Paper: https://arxiv.org/abs/2601.20540"
