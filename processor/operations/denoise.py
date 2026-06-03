"""
operations/denoise.py — Denoise images (OpenCV) and video (FFmpeg nlmeans/hqdn3d).
"""

import logging
import shutil
import subprocess

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _run_logged(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        log.error("Command failed: %s", " ".join(str(c) for c in cmd))
        if stdout:
            log.error("STDOUT: %s", stdout)
        if stderr:
            log.error("STDERR: %s", stderr)
        raise


def denoise_media(input_path: str, output_path: str,
                  media_type: str = "image", settings: dict = None) -> None:
    settings = settings or {}
    strength = settings.get("denoise_strength", "medium")  # low / medium / high

    if media_type == "image":
        _denoise_image(input_path, output_path, strength)
    else:
        _denoise_video(input_path, output_path, strength)


# ── Image ─────────────────────────────────────────────────────────────────────
def _denoise_image(input_path: str, output_path: str, strength: str) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        shutil.copy2(input_path, output_path)
        return

    h_luma   = {"low": 3,  "medium": 7,  "high": 14}[strength]
    h_color  = {"low": 3,  "medium": 7,  "high": 12}[strength]
    template = {"low": 7,  "medium": 7,  "high": 7 }[strength]
    search   = {"low": 21, "medium": 21, "high": 35}[strength]

    if len(img.shape) == 2 or img.shape[2] == 1:
        denoised = cv2.fastNlMeansDenoising(img, None, h_luma, template, search)
    else:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h_luma, h_color, template, search)

    cv2.imwrite(output_path, denoised)
    log.info("Image denoise done -> %s", output_path)


# ── Video ─────────────────────────────────────────────────────────────────────
def _denoise_video(input_path: str, output_path: str, strength: str) -> None:
    """
    Use FFmpeg's nlmeans (high quality) or hqdn3d (fast) filter.
    hqdn3d params: luma_spatial:chroma_spatial:luma_tmp:chroma_tmp
    """
    filters = {
        "low":    "hqdn3d=1.5:1.5:6:6",
        "medium": "hqdn3d=3:3:6:6",
        "high":   "nlmeans=s=6:p=7:r=15",   # slower but better
    }
    vf = filters.get(strength, filters["medium"])

    _run_logged([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "copy",
        output_path,
    ])
    log.info("Video denoise done -> %s", output_path)
