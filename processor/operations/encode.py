"""
operations/encode.py — Final video re-encode with GPU acceleration when available.
Tries NVENC (NVIDIA) → VAAPI (AMD) → libx264 (CPU) in order.
"""

import logging
import shutil
import subprocess

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


def reencode_video(input_path: str, output_path: str, settings: dict = None) -> None:
    if input_path == output_path:
        return
    settings = settings or {}
    crf      = str(settings.get("crf", 18))
    preset   = settings.get("preset", "medium")

    if _try_nvenc(input_path, output_path, crf):
        return
    if _try_vaapi(input_path, output_path):
        return
    _fallback_cpu(input_path, output_path, crf, preset)


def _try_nvenc(inp, out, crf):
    try:
        _run_logged([
            "ffmpeg", "-y",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", inp,
            "-c:v", "h264_nvenc",
            "-rc", "vbr", "-cq", crf,
            "-preset", "p4",          # NVENC quality preset
            "-c:a", "copy",
            out,
        ])
        log.info("Encoded with NVENC -> %s", out)
        return True
    except subprocess.CalledProcessError:
        log.debug("NVENC not available, trying VAAPI.")
        return False


def _try_vaapi(inp, out):
    try:
        _run_logged([
            "ffmpeg", "-y",
            "-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128",
            "-hwaccel_output_format", "vaapi",
            "-i", inp,
            "-vf", "format=nv12|vaapi,hwupload",
            "-c:v", "h264_vaapi",
            "-qp", "20",
            "-c:a", "copy",
            out,
        ])
        log.info("Encoded with VAAPI -> %s", out)
        return True
    except subprocess.CalledProcessError:
        log.debug("VAAPI not available, falling back to CPU.")
        return False


def _fallback_cpu(inp, out, crf, preset):
    _run_logged([
        "ffmpeg", "-y", "-i", inp,
        "-c:v", "libx264", "-crf", crf, "-preset", preset,
        "-c:a", "copy",
        out,
    ])
    log.info("Encoded with libx264 (CPU) -> %s", out)
