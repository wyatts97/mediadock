"""
operations/crop.py — Remove black bars using FFmpeg cropdetect filter.
"""

import logging
import subprocess
import re

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


def crop_black_bars(input_path: str, output_path: str, bars: dict) -> None:
    """
    Detect and crop black bars.
    If the AI already measured them, use those values; otherwise run cropdetect.
    """
    crop_filter = None

    # Try to use AI-supplied measurements first
    if bars.get("detected"):
        t = bars.get("top_px", 0)
        b = bars.get("bottom_px", 0)
        l = bars.get("left_px", 0)
        r = bars.get("right_px", 0)
        if t or b or l or r:
            # Probe the actual dimensions
            probe = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0", input_path,
            ], stderr=subprocess.DEVNULL).decode().strip().split(",")
            w, h = int(probe[0]), int(probe[1])
            cw = w - l - r
            ch = h - t - b
            cx = l
            cy = t
            crop_filter = f"crop={cw}:{ch}:{cx}:{cy}"
            log.info("Using AI crop: %s", crop_filter)

    # Fallback: FFmpeg cropdetect auto-scan
    if crop_filter is None:
        log.info("Running FFmpeg cropdetect on %s", input_path)
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-vf", "cropdetect=24:16:0",
             "-vframes", "300", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)
        if crops:
            # Use the most commonly detected crop value
            from collections import Counter
            crop_filter = "crop=" + Counter(crops).most_common(1)[0][0]
            log.info("cropdetect result: %s", crop_filter)
        else:
            log.warning("No black bars detected, copying input.")
            import shutil; shutil.copy2(input_path, output_path)
            return

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", crop_filter,
        "-c:a", "copy",
        output_path,
    ]
    _run_logged(cmd)
    log.info("Crop done -> %s", output_path)
