from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def load_rgb_image(image_path: str | Path, image_size: int | tuple[int, int] | None = None) -> Image.Image:
    """Open an image with PIL, convert it to RGB, and optionally resize it."""
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
    if image_size is None:
        return rgb_image
    return rgb_image.resize(_normalize_size(image_size), resample=Image.Resampling.BICUBIC)


def inspect_image_metadata(image_path: str | Path, image_size: int | tuple[int, int] | None = None) -> dict[str, Any]:
    image = load_rgb_image(image_path, image_size=image_size)
    width, height = image.size
    return {
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "mode": image.mode,
        "reads_image_pixels": True,
        "loads_model": False,
        "extracts_features": False,
        "trains_model": False,
        "evaluates_model": False,
        "is_paper_result": False,
    }


def _normalize_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        return (image_size, image_size)
    if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size must contain two positive dimensions")
    return image_size
