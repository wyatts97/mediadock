"""
operations/upscale.py — AI upscaling via Real-ESRGAN.
Images are upscaled directly. Videos are frame-extracted, upscaled, reassembled.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_DIR   = os.environ.get("MODEL_DIR", "/models/esrgan")
ESRGAN_PATH = "/opt/Real-ESRGAN/inference_realesrgan.py"


def _run_logged(cmd, cwd=None):
    """Run a subprocess and log stdout/stderr if it fails."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        log.error("Command failed: %s", " ".join(str(c) for c in cmd))
        if stdout:
            log.error("STDOUT: %s", stdout)
        if stderr:
            log.error("STDERR: %s", stderr)
        raise

# Model name → weights file
MODELS = {
    "2x":    ("RealESRGAN_x2plus",      f"{MODEL_DIR}/RealESRGAN_x2plus.pth",      2),
    "4x":    ("RealESRGAN_x4plus",      f"{MODEL_DIR}/RealESRGAN_x4plus.pth",      4),
    "4x_anime": ("RealESRGAN_x4plus_anime_6B", f"{MODEL_DIR}/RealESRGAN_x4plus_anime_6B.pth", 4),
}


def upscale_media(input_path: str, output_path: str,
                  factor: str = "2x", media_type: str = "image",
                  settings: dict = None) -> None:
    settings = settings or {}
    model_key = settings.get("esrgan_model", factor)
    if model_key not in MODELS:
        model_key = "2x"

    model_name, model_path, scale = MODELS[model_key]

    if media_type == "image":
        _upscale_image(input_path, output_path, model_name, model_path, scale)
    else:
        _upscale_video(input_path, output_path, model_name, model_path, scale)


def _upscale_image(input_path, output_path, model_name, model_path, scale):
    """Upscale a single image using a temp dir to avoid overwriting the input."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_input = os.path.join(tmp_dir, Path(input_path).name)
        shutil.copy2(input_path, tmp_input)
        cmd = [
            "python3", ESRGAN_PATH,
            "-i", tmp_input,
            "-o", tmp_dir,
            "-n", model_name,
            "--model_path", model_path,
            "--outscale", str(scale),
            "--fp32",          # CPU-safe full precision
            "--suffix", "",    # preserve original filename
        ]
        _run_logged(cmd, cwd="/opt/Real-ESRGAN")
        result = os.path.join(tmp_dir, Path(input_path).name)
        shutil.move(result, output_path)
    log.info("Image upscale done -> %s", output_path)


def _upscale_video(input_path, output_path, model_name, model_path, scale):
    """Extract frames → upscale each → re-encode with original audio."""
    with tempfile.TemporaryDirectory() as frames_dir, \
         tempfile.TemporaryDirectory() as up_dir:

        frames_in  = os.path.join(frames_dir, "frame_%06d.png")
        frames_out = up_dir   # ESRGAN outputs here

        # 1. Extract frames
        log.info("Extracting video frames…")
        _run_logged(
            ["ffmpeg", "-i", input_path, "-q:v", "1", frames_in],
        )

        # 2. Upscale frames
        log.info("Upscaling frames with %s…", model_name)
        _run_logged([
            "python3", ESRGAN_PATH,
            "-i", frames_dir,
            "-o", up_dir,
            "-n", model_name,
            "--model_path", model_path,
            "--outscale", str(scale),
            "--fp32",
            "--suffix", "",
        ], cwd="/opt/Real-ESRGAN")

        # 3. Get original FPS
        fps_out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", input_path,
        ], stderr=subprocess.DEVNULL).decode().strip()

        # 4. Reassemble with original audio
        log.info("Reassembling upscaled video…")
        _run_logged([
            "ffmpeg", "-y",
            "-framerate", fps_out,
            "-i", os.path.join(up_dir, "frame_%06d.png"),
            "-i", input_path,
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-c:a", "copy",
            output_path,
        ])

    log.info("Video upscale done -> %s", output_path)
