"""
operations/pipeline.py — Orchestrates the full edit pipeline for a job.
"""

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from utils.vision import analyze_media
from operations.crop      import crop_black_bars
from operations.upscale   import upscale_media
from operations.denoise   import denoise_media
from operations.color     import color_correct
from operations.encode    import reencode_video
from operations.watermark import watermark_stub

log = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/output")


def run_pipeline(job: dict, progress_cb: Optional[Callable] = None) -> dict:
    """
    Execute all requested (or AI-detected) operations in order.
    Returns {"output": path, "operations_run": [...], "plan": {...}}
    """
    def pct(p, msg):
        if progress_cb:
            progress_cb(p, msg)

    input_path  = job["input"]
    output_path = job["output"]
    settings    = job.get("settings", {})
    media_type  = job.get("media_type", "image")

    # ── 1. AI analysis (unless ops already supplied) ──────────
    pct(5, "Analyzing media with AI…")
    if job.get("operations"):
        ops  = job["operations"]
        plan = {}
    else:
        plan = analyze_media(input_path)
        ops  = plan.get("suggested_operations", [])

    # Manual overrides from UI settings
    if settings.get("force_upscale"):
        scale = settings.get("upscale_factor", "2x")
        ops.append(f"upscale_{scale}")
    if settings.get("force_denoise"):
        ops.append("denoise")
    if settings.get("force_crop"):
        ops.append("crop_black_bars")

    ops = list(dict.fromkeys(ops))   # deduplicate, preserve order
    log.info("Ops to run: %s", ops)

    current = input_path
    ops_run = []

    # ── 2. Crop black bars ────────────────────────────────────
    if "crop_black_bars" in ops:
        pct(15, "Cropping black bars…")
        out = _stage_path(output_path, "crop")
        crop_black_bars(current, out, plan.get("black_bars", {}))
        current = out
        ops_run.append("crop_black_bars")

    # ── 3. Denoise ────────────────────────────────────────────
    if "denoise" in ops:
        pct(30, "Denoising…")
        out = _stage_path(output_path, "denoise")
        denoise_media(current, out, media_type=media_type, settings=settings)
        current = out
        ops_run.append("denoise")

    # ── 4. Upscale ────────────────────────────────────────────
    scale_op = next((o for o in ops if o.startswith("upscale_")), None)
    if scale_op:
        factor = scale_op.split("_")[-1]   # "2x" or "4x"
        pct(50, f"Upscaling {factor}…")
        out = _stage_path(output_path, f"upscale_{factor}")
        upscale_media(current, out, factor=factor, media_type=media_type, settings=settings)
        current = out
        ops_run.append(scale_op)

    # ── 5. Color correction ───────────────────────────────────
    if "color_correct" in ops or settings.get("force_color"):
        pct(70, "Color correcting…")
        out = _stage_path(output_path, "color")
        color_correct(current, out, media_type=media_type)
        current = out
        ops_run.append("color_correct")

    # ── 6. Watermark stub (no-op placeholder) ─────────────────
    if "watermark_remove_stub" in ops or settings.get("watermark_stub"):
        pct(80, "Watermark removal (stub — not yet implemented)…")
        out = _stage_path(output_path, "wm_stub")
        watermark_stub(current, out)
        current = out
        ops_run.append("watermark_remove_stub")

    # ── 7. Final encode / copy ────────────────────────────────
    pct(90, "Final encode…")
    if media_type == "video":
        reencode_video(current, output_path, settings=settings)
    else:
        import shutil
        shutil.copy2(current, output_path)

    # Clean up stage files
    _cleanup_stages(input_path, output_path, current)

    pct(100, "Done")
    return {
        "output":         output_path,
        "operations_run": ops_run,
        "plan":           plan,
    }


def _stage_path(final_output: str, stage: str) -> str:
    p = Path(final_output)
    return str(p.parent / f"_stage_{stage}{p.suffix}")


def _cleanup_stages(input_path: str, output_path: str, final_stage: str):
    """Remove intermediate stage files."""
    out_dir = Path(output_path).parent
    for f in out_dir.glob("_stage_*"):
        try:
            f.unlink()
        except Exception:
            pass
