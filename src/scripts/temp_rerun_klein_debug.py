"""Temp: rerun Klein on a saved debug folder's input and prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from framework.inference import ModelPool
from framework.inference.flux2_klein.utils import generate_klein_edit
from framework.utils.project import load_project_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="debug folder containing klein_input.png and prompt.txt")
    args = parser.parse_args()

    load_project_environment()
    folder = args.folder
    model_input = Image.open(folder / "klein_input.png").convert("RGB")
    prompt = (folder / "prompt.txt").read_text(encoding="utf-8")

    pool = ModelPool()
    out = generate_klein_edit(pool.flux2_klein_pipeline, model_input, prompt, pool.config.klein)
    out_path = folder / "klein_rerun.png"
    out.save(out_path)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
