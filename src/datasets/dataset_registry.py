from __future__ import annotations

from pathlib import Path

from src.datasets.base_dataset import DatasetDescriptor


_REGISTRY: dict[str, DatasetDescriptor] = {
    "eurosat": DatasetDescriptor("eurosat", "EuroSAT", class_folder_candidates=("images", "RGB", ".")),
    "aid": DatasetDescriptor("aid", "AID", class_folder_candidates=("AID", "images", ".")),
    "nwpu_resisc45": DatasetDescriptor(
        "nwpu_resisc45",
        "NWPU-RESISC45",
        class_folder_candidates=("NWPU-RESISC45", "images", "."),
    ),
}


def get_dataset_descriptor(name: str, root: str | Path | None = None) -> DatasetDescriptor:
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    descriptor = _REGISTRY[key]
    return descriptor.with_root(root) if root is not None else descriptor


def registered_datasets() -> list[str]:
    return sorted(_REGISTRY)
