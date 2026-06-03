"""
operations/upscale.py — AI upscaling via Real-ESRGAN.
Images are upscaled directly. Videos are frame-extracted, upscaled, reassembled.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_DIR   = os.environ.get("MODEL_DIR", "/models/esrgan")
ESRGAN_PATH = "/opt/Real-ESRGAN/inference_realesrgan.py"

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
    out_dir = Path(output_path).parent
    cmd = [
        "python3", ESRGAN_PATH,
        "-i", input_path,
        "-o", str(out_dir),
        "-n", model_name,
        "--model_path", model_path,
        "--outscale", str(scale),
        "--fp16",                    # half-precision for speed
    ]
    subprocess.run(cmd, check=True, capture_output=True, cwd="/opt/Real-ESRGAN")

    # Real-ESRGAN appends model name to filename, rename to expected output
    stem  = Path(input_path).stem
    ext   = Path(output_path).suffix or ".png"
    guess = out_dir / f"{stem}_{model_name}{ext}"
    if guess.exists() and str(guess) != output_path:
        guess.rename(output_path)
    log.info("Image upscale done -> %s", output_path)


def _upscale_video(input_path, output_path, model_name, model_path, scale):
    """Extract frames → upscale each → re-encode with original audio."""
    with tempfile.TemporaryDirectory() as frames_dir, \
         tempfile.TemporaryDirectory() as up_dir:

        frames_in  = os.path.join(frames_dir, "frame_%06d.png")
        frames_out = up_dir   # ESRGAN outputs here

        # 1. Extract frames
        log.info("Extracting video frames…")
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-q:v", "1", frames_in],
            check=True, capture_output=True,
        )

        # 2. Upscale frames
        log.info("Upscaling frames with %s…", model_name)
        subprocess.run([
            "python3", ESRGAN_PATH,
            "-i", frames_dir,
            "-o", up_dir,
            "-n", model_name,
            "--model_path", model_path,
            "--outscale", str(scale),
            "--fp16",
        ], check=True, capture_output=True, cwd="/opt/Real-ESRGAN")

        # 3. Get original FPS
        fps_out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", input_path,
        ], stderr=subprocess.DEVNULL).decode().strip()

        # 4. Reassemble with original audio
        log.info("Reassembling upscaled video…")
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", fps_out,
            "-i", os.path.join(up_dir, "frame_%06d.png"),
            "-i", input_path,
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-c:a", "copy",
            output_path,
        ], check=True, capture_output=True)

    log.info("Video upscale done -> %s", output_path)
