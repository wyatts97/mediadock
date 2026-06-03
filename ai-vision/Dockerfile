# ── AI Vision Service ─────────────────────────────────────────
# Runs Ollama serving LLaVA (multimodal) for frame analysis.
# On first boot it pulls the model automatically.
# NVIDIA build; for AMD swap the base image tag and remove CUDA env vars.

FROM ollama/ollama:latest

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Startup script: launch ollama then pull model if not cached
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV OLLAMA_MODELS=/models/ollama
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_KEEP_ALIVE=24h
ENV OLLAMA_NUM_PARALLEL=2

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
