"""Replace an object by removing it with Klein, then adding the new object into the same mask."""

from __future__ import annotations

from datetime import datetime

from PIL import Image

from ..pool import ModelPool
from .add_object import flux2_klein_add_object_with_box
from .remove_object import flux2_klein_remove_object
from .replace_object import KLEIN_DEBUG_DIR


def flux2_klein_replace_object_two_stage(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
) -> Image.Image:
    """Remove the object at ``mask``, then add ``prompt`` inside its bounding box."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")

    removed = flux2_klein_remove_object(pool, image, mask)
    left, top, right, bottom = mask.convert("L").getbbox()
    instruction = (
        f"Place {prompt} to the center, matching the size of the area it occupies. "
        "Preserve the surroundings."
    )
    result = flux2_klein_add_object_with_box(
        pool, removed, (left, top, right, bottom), prompt, instruction=instruction
    )

    folder = KLEIN_DEBUG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=True)
    removed.save(folder / "removed.png")
    result.save(folder / "result.png")
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"[replace_object two_stage] debug saved to {folder}", flush=True)
    return result
