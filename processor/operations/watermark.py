"""
operations/watermark.py — STUB: Watermark removal placeholder.

This is intentionally a no-op pass-through.
Future implementation will use IOPaint (LaMa model) for inpainting.

Planned approach:
  1. LLaVA detects watermark bounding box(es)
  2. SAM2 generates a precise mask
  3. LaMa inpaints the masked region
  4. For video: apply per-frame, blend with optical-flow for temporal consistency
"""

import logging
import shutil

log = logging.getLogger(__name__)


def watermark_stub(input_path: str, output_path: str) -> None:
    """
    STUB — copies input unchanged.
    Logs a notice so the user knows it's a placeholder.
    """
    log.warning(
        "[STUB] Watermark removal not yet implemented. "
        "Passing through unchanged: %s -> %s",
        input_path, output_path,
    )
    shutil.copy2(input_path, output_path)
