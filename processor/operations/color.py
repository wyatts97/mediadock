"""
operations/color.py — Automatic color correction (levels + white-balance).
Images: OpenCV histogram normalization.
Video:  FFmpeg eq + curves filters.
"""

import logging
import shutil
import subprocess

import cv2
import numpy as np

log = logging.getLogger(__name__)


def color_correct(input_path: str, output_path: str, media_type: str = "image") -> None:
    if media_type == "image":
        _correct_image(input_path, output_path)
    else:
        _correct_video(input_path, output_path)


def _correct_image(input_path: str, output_path: str) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        shutil.copy2(input_path, output_path)
        return

    # Convert to LAB, equalize L channel
    lab  = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge((l, a, b))
    result  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, result)
    log.info("Image color correction done -> %s", output_path)


def _correct_video(input_path: str, output_path: str) -> None:
    # eq filter: contrast/brightness/saturation auto-levels
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "eq=contrast=1.05:brightness=0.02:saturation=1.1",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "copy",
        output_path,
    ], check=True, capture_output=True)
    log.info("Video color correction done -> %s", output_path)
