#!/usr/bin/env python3
"""Temporary two-pass FLUX Fill edit on Replicate (sofa first, then baubles).

Usage:
    uv run src/scripts/temp_flux_fill_two_pass.py --image room.png \
        --sofa-mask sofa_mask.png --bauble-mask bauble_mask.png --output-dir .out/temp_fill

Requires REPLICATE_API_TOKEN. White pixels in a mask are repainted.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import replicate
import requests
from PIL import Image
from replicate_edit_one import TERMINAL_STATUSES, output_image_url

MODEL = "black-forest-labs/flux-fill-dev"
SOFA_PROMPT = (
    "Reupholster the masked sofa in dusty-pink, soft fabric matching the pale-pink bauble beside it."
)
BAUBLE_PROMPT = "Replace the masked baubles with small red-and-white Christmas stockings."


def load_png_bytes(path: Path, size: tuple[int, int] | None = None, mode: str = "RGB") -> bytes:
    image = Image.open(path).convert(mode)
    if size is not None and image.size != size:
        resample = Image.Resampling.NEAREST if mode == "L" else Image.Resampling.LANCZOS
        image = image.resize(size, resample)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def download_image(url: str, attempts: int = 3) -> Image.Image:
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=(20, 120))
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                return image.convert("RGB")
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def run_fill(
    client: replicate.Client,
    image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    seed: int,
    poll_seconds: float,
    max_wait: float,
    record_path: Path,
) -> Image.Image:
    prediction = client.predictions.create(
        model=MODEL,
        input={
            "image": io.BytesIO(image_bytes),
            "mask": io.BytesIO(mask_bytes),
            "prompt": prompt,
            "seed": seed,
            "output_format": "png",
        },
    )
    print(f"Created prediction {prediction.id}, waiting...", flush=True)

    started = time.monotonic()
    while True:
        prediction = client.predictions.get(prediction.id)
        if prediction.status in TERMINAL_STATUSES:
            break
        if time.monotonic() - started > max_wait:
            raise TimeoutError(f"Prediction {prediction.id} did not finish within {max_wait}s.")
        time.sleep(poll_seconds)

    if prediction.status != "succeeded":
        raise RuntimeError(f"Prediction {prediction.status}: {prediction.error}")

    url = output_image_url(prediction.output)
    record_path.write_text(json.dumps({"id": prediction.id, "url": url}, indent=2), encoding="utf-8")
    return download_image(url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--sofa-mask", required=True, type=Path)
    parser.add_argument("--bauble-mask", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".out/temp_flux_fill"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-wait", type=float, default=1200.0)
    args = parser.parse_args()

    for path in (args.image, args.sofa_mask, args.bauble_mask):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise SystemExit("Set REPLICATE_API_TOKEN before running.")

    size = Image.open(args.image).size
    client = replicate.Client(api_token=token)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = load_png_bytes(args.image)
    sofa_mask = load_png_bytes(args.sofa_mask, size, "L")
    bauble_mask = load_png_bytes(args.bauble_mask, size, "L")

    pass1 = run_fill(
        client,
        image_bytes,
        sofa_mask,
        SOFA_PROMPT,
        args.seed,
        args.poll_seconds,
        args.max_wait,
        args.output_dir / "pass1_prediction.json",
    )
    if pass1.size != size:
        pass1 = pass1.resize(size, Image.Resampling.LANCZOS)
    pass1_path = args.output_dir / "pass1_sofa.png"
    pass1.save(pass1_path)
    print(f"Saved: {pass1_path}", flush=True)

    pass2 = run_fill(
        client,
        load_png_bytes(pass1_path),
        bauble_mask,
        BAUBLE_PROMPT,
        args.seed,
        args.poll_seconds,
        args.max_wait,
        args.output_dir / "pass2_prediction.json",
    )
    if pass2.size != size:
        pass2 = pass2.resize(size, Image.Resampling.LANCZOS)
    pass2_path = args.output_dir / "pass2_baubles.png"
    pass2.save(pass2_path)
    print(f"Saved: {pass2_path}")


if __name__ == "__main__":
    main()
