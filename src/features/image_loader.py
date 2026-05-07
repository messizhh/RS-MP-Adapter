from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import read_json


DEFAULT_SPLIT_SECTIONS = ("support", "train", "val", "test")


@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    relative_path: str
    label: int
    class_name: str
    section: str
    sample_index: int


def load_split_samples(
    split_path: str | Path,
    dataset_root: str | Path,
    sections: Iterable[str] = DEFAULT_SPLIT_SECTIONS,
    max_samples: int | None = None,
) -> list[ImageSample]:
    """Load split entries as resolved image paths without opening images or models."""
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")

    split = read_json(split_path)
    root = Path(dataset_root)
    samples: list[ImageSample] = []
    for section in sections:
        section_samples = split.get(section, [])
        if not isinstance(section_samples, list):
            raise ValueError(f"Split section is not a list: {section}")
        for sample_index, raw_sample in enumerate(section_samples):
            if max_samples is not None and len(samples) >= max_samples:
                return samples
            samples.append(_parse_sample(root, section, sample_index, raw_sample))
    return samples


def resolve_relative_image_path(dataset_root: str | Path, sample_path: str) -> Path:
    if not isinstance(sample_path, str) or not sample_path:
        raise ValueError("sample path must be a non-empty string")

    relative_path = Path(sample_path)
    if relative_path.is_absolute():
        raise ValueError("absolute sample paths are not allowed")

    root = Path(dataset_root)
    image_path = root / relative_path
    resolved_root = root.resolve(strict=False)
    resolved_image_path = image_path.resolve(strict=False)
    if not resolved_image_path.is_relative_to(resolved_root):
        raise ValueError("sample path escapes dataset root")
    return image_path


def image_loading_safety_metadata() -> dict[str, bool]:
    return {
        "is_paper_result": False,
        "loads_model": False,
        "extracts_features": False,
        "trains_model": False,
        "evaluates_model": False,
    }


def _parse_sample(dataset_root: Path, section: str, sample_index: int, raw_sample: Any) -> ImageSample:
    if not isinstance(raw_sample, dict):
        raise ValueError(f"Split sample is not an object: {section}[{sample_index}]")

    image_path = resolve_relative_image_path(dataset_root, raw_sample.get("path"))
    label = raw_sample.get("label")
    class_name = raw_sample.get("class_name")
    if not isinstance(label, int):
        raise ValueError(f"Split sample label is not an integer: {section}[{sample_index}]")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError(f"Split sample class_name is missing: {section}[{sample_index}]")

    return ImageSample(
        image_path=image_path,
        relative_path=raw_sample["path"],
        label=label,
        class_name=class_name,
        section=section,
        sample_index=sample_index,
    )
