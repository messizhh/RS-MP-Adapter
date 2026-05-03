from __future__ import annotations

from pathlib import Path


def create_fake_class_dataset(
    root: Path,
    class_names: list[str] | None = None,
    samples_per_class: int = 20,
    nested_root: str | None = None,
    include_hidden: bool = False,
    include_unsupported: bool = False,
    empty_class: str | None = None,
) -> Path:
    class_root = root / nested_root if nested_root else root
    class_root.mkdir(parents=True, exist_ok=True)
    names = class_names or ["class_0", "class_1", "class_2"]
    for class_name in names:
        class_dir = class_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        if class_name == empty_class:
            continue
        for index in range(samples_per_class):
            (class_dir / f"sample_{index:03d}.jpg").write_text("fake image placeholder\n", encoding="utf-8")
        if include_hidden:
            (class_dir / ".hidden.jpg").write_text("hidden placeholder\n", encoding="utf-8")
            hidden_dir = class_dir / ".hidden_dir"
            hidden_dir.mkdir()
            (hidden_dir / "nested.jpg").write_text("hidden nested placeholder\n", encoding="utf-8")
        if include_unsupported:
            (class_dir / "notes.txt").write_text("unsupported placeholder\n", encoding="utf-8")
    return class_root


def create_fake_eurosat(root: Path, samples_per_class: int = 20) -> Path:
    return create_fake_class_dataset(root, [f"eurosat_class_{idx}" for idx in range(3)], samples_per_class, nested_root="RGB")


def create_fake_aid(root: Path, samples_per_class: int = 20) -> Path:
    return create_fake_class_dataset(root, [f"aid_class_{idx}" for idx in range(3)], samples_per_class, nested_root="AID")


def create_fake_nwpu(root: Path, samples_per_class: int = 20) -> Path:
    return create_fake_class_dataset(root, [f"nwpu_class_{idx}" for idx in range(3)], samples_per_class, nested_root="NWPU-RESISC45")
