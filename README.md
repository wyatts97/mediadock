# MEDIA//AI — AI-Powered Media Processing Engine

Drag-drop video/image processor backed by a local LLaVA vision model.
No cloud. No censorship. Runs entirely on your hardware.

---

## Stack

| Service      | Role                                      | Internal Port |
|--------------|-------------------------------------------|---------------|
| `ai-vision`  | Ollama + LLaVA (multimodal analysis)      | 11434         |
| `processor`  | Celery worker: FFmpeg, Real-ESRGAN, etc.  | —             |
| `api`        | FastAPI job queue + REST endpoints        | 8317          |
| `web`        | Nginx + Web UI                            | **4953→7841** |
| `redis`      | Job queue + result backend                | (internal)    |

**Single outward port: `4953`** → `http://localhost:4953`

---

## Quick Start

### Prerequisites

```bash
# NVIDIA
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# AMD — install ROCm:
# https://rocm.docs.amd.com/en/latest/deploy/linux/index.html
```

### Run

```bash
git clone <this-repo> media-ai
cd media-ai
chmod +x scripts/start.sh
./scripts/start.sh
```

Or manually:

```bash
# NVIDIA
docker compose up -d

# AMD
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d

# CPU only
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

Open **http://localhost:4953** in your browser.

> ⏳ **First boot**: `ai-vision` will download the LLaVA model (~8 GB for 13B, ~4 GB for 7B).
> Monitor with: `docker compose logs -f ai-vision`

---

## Operations

| Operation          | Status | Engine                        |
|--------------------|--------|-------------------------------|
| Black bar removal  | ✅     | FFmpeg `cropdetect`           |
| Upscale 2×/4×      | ✅     | Real-ESRGAN x2plus / x4plus  |
| Anime upscale      | ✅     | Real-ESRGAN x4plus anime 6B  |
| Denoise (image)    | ✅     | OpenCV `fastNlMeans`          |
| Denoise (video)    | ✅     | FFmpeg `hqdn3d` / `nlmeans`   |
| Color correction   | ✅     | OpenCV CLAHE + FFmpeg eq      |
| Watermark removal  | 🚧     | **Stub** (pass-through)       |
| GPU encode         | ✅     | NVENC → VAAPI → libx264       |

---

## GPU Tuning

### Model selection (VRAM)

Edit `ai-vision/entrypoint.sh` or set env:

```yaml
# docker-compose.yml → ai-vision
environment:
  - LLAVA_MODEL=llava:7b      # ~4 GB VRAM
  - LLAVA_MODEL=llava:13b     # ~8 GB VRAM  (default)
  - LLAVA_MODEL=llava:34b     # ~20 GB VRAM (best quality)
```

### AMD GFX version

Edit `docker-compose.amd.yml`:
```yaml
- HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6000 series
- HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7000 series
- HSA_OVERRIDE_GFX_VERSION=9.0.6    # Vega / RX 5000
```

---

## File Limits

Upload limit is set to **10 GB** in Nginx. Adjust in `web/nginx.conf`:
```nginx
client_max_body_size 10g;
```

---

## Watermark Removal (Future)

The stub is in `processor/operations/watermark.py`.
Planned implementation:
1. LLaVA detects bounding box
2. SAM2 generates precise mask
3. LaMa / IOPaint inpaints
4. Video: per-frame + optical-flow temporal blending

---

## Stop / Clean up

```bash
docker compose down          # stop, keep volumes
docker compose down -v       # stop + delete volumes (clears models!)
```
