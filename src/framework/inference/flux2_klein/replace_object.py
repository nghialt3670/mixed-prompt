"""Replace an object with a new one in a single FLUX.2 Klein pass."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..pool import ModelPool
from .placement import GUIDE
from .utils import (
    WORK_SIZE,
    binary_mask,
    generate_klein_edit,
)

KLEIN_DEBUG_DIR = Path(".out/klein_debug")
REPLACE_PROMPT = "Replace the object in the middle of the image with {prompt}. {guide} Add nothing else and preserve the surroundings."
# The old object's area (grown by this fraction of its size) takes Klein's pixels.
OLD_AREA_GROW = 0.04


def _old_area_weight(mask: Image.Image, crop_box) -> np.ndarray:
    """Soft weight on the old object's footprint, so none of it is left behind."""
    left, top, right, bottom = mask.getbbox()
    grow = max(2, round(OLD_AREA_GROW * max(right - left, bottom - top)))
    grown = mask.crop(crop_box).filter(ImageFilter.MaxFilter(2 * grow + 1))
    soft = grown.filter(ImageFilter.GaussianBlur(grow / 2))
    return np.asarray(soft, np.float32) / 255.0


def _save_klein_debug(model_input: Image.Image, model_output: Image.Image, prompt: str) -> Path:
    """Write Klein's input, output, and prompt to a timestamped folder under .out/klein_debug."""
    folder = KLEIN_DEBUG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=True)
    model_input.save(folder / "klein_input.png")
    model_output.save(folder / "klein_output.png")
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"[replace_object] klein debug saved to {folder}", flush=True)
    return folder


def flux2_klein_replace_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    debug: dict | None = None,
    *,
    refine: bool = False,
) -> Image.Image:
    """Replace the object at a white mask with ``prompt``.

    Klein's edit of the crop is pasted directly over the mask's soft footprint.
    """
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("image and mask must both be PIL images")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if mask.size != image.size:
        raise ValueError("mask must have the same size as image")
    original = image.convert("RGB")
    mask = Image.fromarray(binary_mask(mask).astype(np.uint8) * 255, "L")
    region = mask.getbbox()
    if region is None:
        raise ValueError("mask is empty")
    inference_config = pool.config.klein
    left, top, right, bottom = region
    padding = inference_config.mask_context_padding
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(original.width, right + padding),
        min(original.height, bottom + padding),
    )
    original_crop = original.crop(crop_box)
    working_image = ImageOps.contain(original_crop, (WORK_SIZE, WORK_SIZE))

    full_prompt = REPLACE_PROMPT.format(prompt=prompt, guide=GUIDE)
    out = generate_klein_edit(
        pool.flux2_klein_pipeline,
        working_image,
        full_prompt,
        inference_config,
    )
    if out.size != working_image.size:
        out = out.resize(working_image.size, Image.Resampling.LANCZOS)
    _save_klein_debug(working_image, out, full_prompt)
    edited = out.resize(original_crop.size, Image.Resampling.LANCZOS)
    weight = _old_area_weight(mask, crop_box)
    paste_mask = Image.fromarray((weight * 255).astype(np.uint8), "L")
    full = original.copy()
    full.paste(Image.composite(edited, original_crop, paste_mask), crop_box[:2])
    if debug is not None:
        debug.update(before=original_crop, klein=edited, weight=weight)
    return full
