# ── Media Processor ───────────────────────────────────────────
# FFmpeg (NVENC/VAAPI) + Real-ESRGAN + BasicVSR++ + OpenCV
# NVIDIA base — for AMD replace with rocm/pytorch base and set ROCm env vars.

FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System packages ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg with NVENC/NVDEC support
    ffmpeg \
    # Python
    python3.11 python3.11-dev python3-pip \
    # OpenCV deps
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    # Misc
    wget curl git unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ── Python deps ───────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Real-ESRGAN ───────────────────────────────────────────────
WORKDIR /opt
RUN git clone https://github.com/xinntao/Real-ESRGAN.git && \
    cd Real-ESRGAN && \
    pip install --no-cache-dir basicsr facexlib gfpgan && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e . --no-deps

# Download ESRGAN weights
RUN mkdir -p /models/esrgan && \
    wget -q -O /models/esrgan/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget -q -O /models/esrgan/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    wget -q -O /models/esrgan/RealESRGAN_x4plus_anime_6B.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

# ── App code ─────────────────────────────────────────────────
WORKDIR /app
COPY worker.py        /app/worker.py
COPY operations/      /app/operations/
COPY utils/           /app/utils/

ENV REDIS_URL=redis://redis:6379/0
ENV AI_VISION_URL=http://ai-vision:11434
ENV MODEL_DIR=/models/esrgan
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output
ENV PYTHONPATH=/opt/Real-ESRGAN:/app

CMD ["python3", "worker.py"]
