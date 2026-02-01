# LingBot-World RunPod Deployment
# Requires: A100-80GB GPU

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    packaging \
    wheel \
    setuptools \
    ninja

# Install PyTorch and flash-attn
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# Install flash-attn (requires CUDA compiler)
RUN pip install flash-attn --no-build-isolation

# Install remaining dependencies
RUN pip install --no-cache-dir \
    diffusers>=0.31.0 \
    "transformers>=4.49.0,<=4.51.3" \
    tokenizers>=0.20.3 \
    accelerate>=1.1.1 \
    opencv-python>=4.9.0.80 \
    "imageio[ffmpeg]" \
    imageio-ffmpeg \
    "Pillow>=10.0.0" \
    tqdm \
    easydict \
    ftfy \
    "einops>=0.7.0" \
    "numpy>=1.23.5,<2" \
    scipy \
    safetensors>=0.4.0 \
    huggingface-hub>=0.20.0 \
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    pydantic>=2.5.0 \
    python-multipart

# Clone LingBot-World repository
RUN cd /workspace && git clone https://github.com/Robbyant/lingbot-world.git
RUN cd /workspace/lingbot-world && pip install -e .

# Copy server code
COPY server.py /workspace/server.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Expose port
EXPOSE 8000

# Start command
CMD ["/workspace/start.sh"]
