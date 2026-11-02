"""Download a Replicate result URL with a long timeout and retries."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

CONNECT_TIMEOUT_S = 60
READ_TIMEOUT_S = 900


def download(url: str, output: Path, attempts: int) -> None:
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) as response:
                response.raise_for_status()
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        handle.write(chunk)
            print(f"saved {output}")
            return
        except requests.RequestException as exc:
            print(f"attempt {attempt}/{attempts} failed: {exc}")
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--output", type=Path, default=Path(".out/temp_flux_fill/result.jpg"))
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    download(args.url, args.output, args.attempts)


if __name__ == "__main__":
    main()
