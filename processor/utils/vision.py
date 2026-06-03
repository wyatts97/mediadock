"""
utils/vision.py — Sends a representative frame to LLaVA for analysis.
Returns a structured EditPlan dict.
"""

import base64
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

AI_VISION_URL = os.environ.get("AI_VISION_URL", "http://ai-vision:11434")
LLAVA_MODEL   = os.environ.get("LLAVA_MODEL", "llava:13b")

ANALYSIS_PROMPT = """You are a professional video and image editor AI.
Analyze the provided media frame and return ONLY a valid JSON object — no markdown, no explanation.

JSON schema:
{
  "black_bars": {
    "detected": false,
    "top_px": 0,
    "bottom_px": 0,
    "left_px": 0,
    "right_px": 0
  },
  "resolution_quality": "poor|acceptable|good",
  "needs_upscale": false,
  "recommended_upscale": "2x|4x|none",
  "noise_level": "low|medium|high",
  "color_issues": [],
  "deinterlace_needed": false,
  "fps_looks_low": false,
  "suggested_operations": [],
  "confidence": 0.0
}

Possible suggested_operations values:
  "crop_black_bars", "upscale_2x", "upscale_4x", "denoise",
  "color_correct", "deinterlace", "sharpen", "watermark_remove_stub"

Be conservative. Only suggest operations where there is clear visible need.
Return ONLY the JSON object."""


def _extract_frame(input_path: str, timestamp: float = 0.0) -> bytes:
    """Extract a single frame from a video or return the image itself."""
    suffix = Path(input_path).suffix.lower()
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

    if suffix in image_exts:
        with open(input_path, "rb") as f:
            return f.read()

    # Video: grab a frame near the 10% mark
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_path,
    ]
    try:
        duration = float(subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip())
        timestamp = max(0.0, duration * 0.10)
    except Exception:
        timestamp = 3.0

    subprocess.run(
        ["ffmpeg", "-ss", str(timestamp), "-i", input_path,
         "-vframes", "1", "-q:v", "2", "-y", tmp_path],
        check=True, capture_output=True,
    )
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


def analyze_media(input_path: str) -> dict:
    """
    Analyze media file with LLaVA. Returns an EditPlan dict.
    Falls back to a safe default plan on any error.
    """
    try:
        frame_bytes = _extract_frame(input_path)
        b64_image   = base64.b64encode(frame_bytes).decode()

        payload = {
            "model": LLAVA_MODEL,
            "prompt": ANALYSIS_PROMPT,
            "images": [b64_image],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512},
        }

        resp = httpx.post(
            f"{AI_VISION_URL}/api/generate",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        plan = json.loads(raw.strip())
        log.info("LLaVA analysis done: ops=%s", plan.get("suggested_operations"))
        return plan

    except Exception as exc:
        log.warning("Vision analysis failed (%s), using safe defaults.", exc)
        return _safe_defaults()


def _safe_defaults() -> dict:
    return {
        "black_bars":           {"detected": False, "top_px": 0, "bottom_px": 0, "left_px": 0, "right_px": 0},
        "resolution_quality":   "acceptable",
        "needs_upscale":        False,
        "recommended_upscale":  "none",
        "noise_level":          "low",
        "color_issues":         [],
        "deinterlace_needed":   False,
        "fps_looks_low":        False,
        "suggested_operations": [],
        "confidence":           0.0,
    }
