#!/usr/bin/env bash
# ============================================================
#  start.sh — detect GPU(s) and launch the right compose config
# ============================================================
set -e

NVIDIA=false
AMD=false

# Detect NVIDIA
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    NVIDIA=true
    echo "[start] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Detect AMD (ROCm)
if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
    AMD=true
    echo "[start] AMD GPU detected (ROCm)"
fi

if [ "$NVIDIA" = false ] && [ "$AMD" = false ]; then
    echo "[start] WARNING: No GPU detected. Falling back to CPU (upscaling will be slow)."
fi

# Export for compose
export NVIDIA_AVAILABLE=$NVIDIA
export AMD_AVAILABLE=$AMD

echo ""
echo "[start] Launching media-ai stack on port 4953..."
echo "[start] Web UI will be available at:  http://localhost:4953"
echo ""

if [ "$NVIDIA" = true ]; then
    # Full compose with NVIDIA runtime
    docker compose up -d
elif [ "$AMD" = true ]; then
    # Use AMD override
    docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
else
    # CPU-only: remove GPU deploy blocks
    docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
fi

echo ""
echo "[start] Services starting. First run will download the LLaVA model (~8GB)."
echo "[start] Monitor logs:   docker compose logs -f ai-vision"
echo "[start] Stop stack:     docker compose down"
