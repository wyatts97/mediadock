#!/bin/bash
# Start Ollama server in background, wait for it, then pull model if missing.

set -e

MODEL="${LLAVA_MODEL:-llava:7b}"   # override via env; llava:7b for low-VRAM

echo "[ai-vision] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to become ready
echo "[ai-vision] Waiting for Ollama API..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "[ai-vision] Ollama ready."

# Pull the model only if not already downloaded
if ! ollama list | grep -q "^${MODEL}"; then
    echo "[ai-vision] Pulling model: ${MODEL}  (this may take a while on first boot)"
    ollama pull "${MODEL}"
else
    echo "[ai-vision] Model ${MODEL} already cached."
fi

echo "[ai-vision] Vision service ready."
wait "$OLLAMA_PID"
