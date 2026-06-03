"""
api/main.py — FastAPI orchestration backend.
Handles file upload, job dispatch, status polling, and result download.
"""

import os
import uuid
import mimetypes
from pathlib import Path
from typing import Optional

import aiofiles
import httpx
from celery import Celery
from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── Config ────────────────────────────────────────────────────
REDIS_URL      = os.environ.get("REDIS_URL",      "redis://redis:6379/0")
AI_VISION_URL  = os.environ.get("AI_VISION_URL",  "http://ai-vision:11434")
INPUT_DIR      = Path(os.environ.get("INPUT_DIR",  "/data/input"))
OUTPUT_DIR     = Path(os.environ.get("OUTPUT_DIR", "/data/output"))

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Celery client (submit only, no worker here) ──────────────
celery = Celery("api_client", broker=REDIS_URL, backend=REDIS_URL)
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

app = FastAPI(title="media-ai API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


def _media_type(path: Path) -> str:
    return "image" if path.suffix.lower() in IMAGE_EXTS else "video"


def _output_ext(input_path: Path, media_type: str) -> str:
    if media_type == "image":
        return input_path.suffix.lower() or ".png"
    return ".mp4"


# ── Routes ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ai-status")
async def ai_status():
    """Check if LLaVA is loaded and ready."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{AI_VISION_URL}/api/tags")
            models = r.json().get("models", [])
            return {"ready": True, "models": [m["name"] for m in models]}
    except Exception as exc:
        return {"ready": False, "error": str(exc)}


@app.post("/jobs")
async def submit_job(
    file:             UploadFile = File(...),
    force_upscale:    bool       = Form(False),
    upscale_factor:   str        = Form("2x"),
    force_denoise:    bool       = Form(False),
    denoise_strength: str        = Form("medium"),
    force_crop:       bool       = Form(False),
    force_color:      bool       = Form(False),
    watermark_stub:   bool       = Form(False),
    esrgan_model:     str        = Form("auto"),
    auto_analyze:     bool       = Form(True),
):
    job_id   = str(uuid.uuid4())
    suffix   = Path(file.filename).suffix.lower()
    in_path  = INPUT_DIR / f"{job_id}_in{suffix}"
    mtype    = _media_type(in_path)
    out_ext  = _output_ext(in_path, mtype)
    out_path = OUTPUT_DIR / f"{job_id}_out{out_ext}"

    # Save upload
    async with aiofiles.open(in_path, "wb") as f:
        await f.write(await file.read())

    job = {
        "job_id":    job_id,
        "input":     str(in_path),
        "output":    str(out_path),
        "media_type": mtype,
        "operations": [] if auto_analyze else [],   # empty = AI picks
        "settings": {
            "force_upscale":    force_upscale,
            "upscale_factor":   upscale_factor,
            "force_denoise":    force_denoise,
            "denoise_strength": denoise_strength,
            "force_crop":       force_crop,
            "force_color":      force_color,
            "watermark_stub":   watermark_stub,
            "esrgan_model":     esrgan_model if esrgan_model != "auto" else upscale_factor,
            "auto_analyze":     auto_analyze,
        },
    }

    task = celery.send_task("process_media", args=[job], queue="media")
    return {"job_id": job_id, "task_id": task.id, "media_type": mtype}


@app.get("/jobs/{task_id}/status")
def job_status(task_id: str):
    result = AsyncResult(task_id, app=celery)

    if result.state == "PENDING":
        return {"state": "queued",   "progress": 0,   "status": "Waiting in queue…"}
    if result.state == "STARTED":
        return {"state": "started",  "progress": 0,   "status": "Starting…"}
    if result.state == "PROGRESS":
        meta = result.info or {}
        return {"state": "running",  "progress": meta.get("progress", 0),
                "status": meta.get("status", "Processing…")}
    if result.state == "SUCCESS":
        info = result.result or {}
        return {
            "state":          "done",
            "progress":       100,
            "status":         "Complete",
            "operations_run": info.get("operations_run", []),
            "output":         info.get("output", ""),
        }
    if result.state == "FAILURE":
        return {"state": "error", "progress": 0, "status": str(result.info)}

    return {"state": result.state.lower(), "progress": 0, "status": result.state}


@app.get("/jobs/{task_id}/download")
def download_result(task_id: str):
    result = AsyncResult(task_id, app=celery)
    if result.state != "SUCCESS":
        raise HTTPException(status_code=404, detail="Result not ready")

    out_path = Path(result.result.get("output", ""))
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Output file missing")

    media_type, _ = mimetypes.guess_type(str(out_path))
    return FileResponse(
        str(out_path),
        media_type=media_type or "application/octet-stream",
        filename=out_path.name,
    )


@app.get("/jobs")
def list_jobs():
    """Return all output files (simple filesystem scan)."""
    files = []
    for f in OUTPUT_DIR.glob("*_out.*"):
        files.append({"filename": f.name, "size": f.stat().st_size})
    return {"outputs": files}
