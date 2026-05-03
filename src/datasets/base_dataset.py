from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetSample:
    path: str
    label: int
    class_name: str


@dataclass(frozen=True)
class DatasetDescriptor:
    name: str
    display_name: str
    root: Path | None = None
    class_folder_candidates: tuple[str, ...] = (".",)
    image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS

    def with_root(self, root: str | Path) -> "DatasetDescriptor":
        return DatasetDescriptor(
            name=self.name,
            display_name=self.display_name,
            root=Path(root),
            class_folder_candidates=self.class_folder_candidates,
            image_extensions=self.image_extensions,
        )


def find_class_root(descriptor: DatasetDescriptor) -> Path:
    if descriptor.root is None:
        raise FileNotFoundError(f"Dataset root is not configured for {descriptor.name}")
    root = Path(descriptor.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    candidates = [root / candidate for candidate in descriptor.class_folder_candidates]
    for candidate in candidates:
        if candidate.exists() and any(child.is_dir() for child in candidate.iterdir()):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No class-folder layout found for {descriptor.name}. Searched: {searched}")


def list_class_folder_samples(descriptor: DatasetDescriptor) -> tuple[list[DatasetSample], dict[str, int]]:
    class_root = find_class_root(descriptor)
    class_dirs = sorted(path for path in class_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found under {class_root}")

    class_to_idx = {path.name: index for index, path in enumerate(class_dirs)}
    samples: list[DatasetSample] = []
    suffixes = tuple(suffix.lower() for suffix in descriptor.image_extensions)
    for class_dir in class_dirs:
        files = sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
        if not files:
            raise ValueError(f"Class folder contains no supported images: {class_dir}")
        label = class_to_idx[class_dir.name]
        for image_path in files:
            samples.append(DatasetSample(path=str(image_path), label=label, class_name=class_dir.name))
    return samples, class_to_idx
