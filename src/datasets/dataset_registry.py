from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets.base_dataset import DatasetDescriptor, descriptor_from_config, normalize_extensions


_REGISTRY: dict[str, DatasetDescriptor] = {
    "eurosat": DatasetDescriptor(
        "eurosat",
        "EuroSAT",
        class_folder_candidates=("images", "RGB", "."),
        expected_num_classes=10,
        output_split_root="splits/eurosat",
    ),
    "aid": DatasetDescriptor(
        "aid",
        "AID",
        class_folder_candidates=("AID", "images", "."),
        expected_num_classes=30,
        output_split_root="splits/aid",
    ),
    "nwpu_resisc45": DatasetDescriptor(
        "nwpu_resisc45",
        "NWPU-RESISC45",
        class_folder_candidates=("NWPU-RESISC45", "images", "."),
        expected_num_classes=45,
        output_split_root="splits/nwpu_resisc45",
    ),
}


def get_dataset_descriptor(
    name: str,
    root: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> DatasetDescriptor:
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    descriptor = descriptor_from_config(config, dataset_name=key, dataset_root=root) if config is not None else _REGISTRY[key]
    if root is not None:
        descriptor = descriptor.with_root(root)
    return descriptor.with_options(image_extensions=normalize_extensions(descriptor.image_extensions))


def registered_datasets() -> list[str]:
    return sorted(_REGISTRY)
