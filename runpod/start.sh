#!/bin/bash
set -e

echo "=============================================="
echo "🚀 LingBot-World RunPod Server Starting"
echo "=============================================="

# Set model directory
MODEL_DIR="/workspace/models"
MODEL_PATH="$MODEL_DIR/lingbot-world-base-cam"

# Download model if not exists
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo ""
    echo "📥 Downloading LingBot-World model (~50GB)..."
    echo "   This may take 30-60 minutes on first run."
    echo ""
    
    mkdir -p "$MODEL_DIR"
    
    # Download using huggingface-cli
    huggingface-cli download robbyant/lingbot-world-base-cam \
        --local-dir "$MODEL_PATH" \
        --local-dir-use-symlinks False
    
    echo "✅ Model downloaded successfully!"
else
    echo "✅ Model already exists at $MODEL_PATH"
fi

# List model files
echo ""
echo "📂 Model files:"
ls -lh "$MODEL_PATH" | head -20

echo ""
echo "=============================================="
echo "🌍 Starting LingBot-World API Server..."
echo "=============================================="

# Start the FastAPI server
cd /workspace
python server.py
