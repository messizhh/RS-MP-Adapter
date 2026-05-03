from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_UNSET = object()


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
    class_name_mapping: dict[str, str] | None = None
    expected_num_classes: int | None = None
    min_images_per_class: int = 1
    output_split_root: str | None = None
    ignore_hidden_files: bool = True
    follow_symlinks: bool = False

    def with_root(self, root: str | Path) -> "DatasetDescriptor":
        return replace(self, root=Path(root))

    def with_options(
        self,
        *,
        image_extensions: list[str] | tuple[str, ...] | None = None,
        class_folder_candidates: list[str] | tuple[str, ...] | None = None,
        class_name_mapping: dict[str, str] | None = None,
        expected_num_classes: int | None | object = _UNSET,
        min_images_per_class: int | None = None,
        output_split_root: str | None = None,
        ignore_hidden_files: bool | None = None,
        follow_symlinks: bool | None = None,
    ) -> "DatasetDescriptor":
        return replace(
            self,
            image_extensions=normalize_extensions(image_extensions or self.image_extensions),
            class_folder_candidates=tuple(class_folder_candidates or self.class_folder_candidates),
            class_name_mapping=class_name_mapping if class_name_mapping is not None else self.class_name_mapping,
            expected_num_classes=expected_num_classes if expected_num_classes is not _UNSET else self.expected_num_classes,
            min_images_per_class=min_images_per_class if min_images_per_class is not None else self.min_images_per_class,
            output_split_root=output_split_root if output_split_root is not None else self.output_split_root,
            ignore_hidden_files=ignore_hidden_files if ignore_hidden_files is not None else self.ignore_hidden_files,
            follow_symlinks=follow_symlinks if follow_symlinks is not None else self.follow_symlinks,
        )


def descriptor_from_config(config: dict[str, Any], dataset_name: str | None = None, dataset_root: str | Path | None = None) -> DatasetDescriptor:
    dataset_cfg = config.get("dataset", config)
    name = dataset_name or dataset_cfg["name"]
    descriptor = DatasetDescriptor(
        name=name,
        display_name=dataset_cfg.get("display_name", name),
        root=Path(dataset_root or dataset_cfg["root"]) if dataset_root or dataset_cfg.get("root") else None,
        class_folder_candidates=tuple(dataset_cfg.get("class_folder_candidates", ["."])),
        image_extensions=normalize_extensions(dataset_cfg.get("image_extensions", IMAGE_EXTENSIONS)),
        class_name_mapping=dataset_cfg.get("class_name_mapping"),
        expected_num_classes=dataset_cfg.get("expected_num_classes"),
        min_images_per_class=int(dataset_cfg.get("min_images_per_class", 1)),
        output_split_root=dataset_cfg.get("output_split_root"),
        ignore_hidden_files=bool(dataset_cfg.get("ignore_hidden_files", True)),
        follow_symlinks=bool(dataset_cfg.get("follow_symlinks", False)),
    )
    return descriptor


def find_class_root(descriptor: DatasetDescriptor) -> Path:
    if descriptor.root is None:
        raise FileNotFoundError(f"Dataset root is not configured for {descriptor.name}")
    root = Path(descriptor.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    candidates = [root / candidate for candidate in descriptor.class_folder_candidates]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and discover_class_dirs(candidate, descriptor.ignore_hidden_files):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No class-folder layout found for {descriptor.name}. Searched: {searched}")


def discover_class_dirs(class_root: Path, ignore_hidden_files: bool = True) -> list[Path]:
    class_dirs = []
    for path in class_root.iterdir():
        if ignore_hidden_files and is_hidden(path):
            continue
        if path.is_dir():
            class_dirs.append(path)
    return sorted(class_dirs, key=lambda path: path.name.lower())


def inspect_class_folder_dataset(
    descriptor: DatasetDescriptor,
    max_classes: int | None = None,
    max_samples_per_class: int | None = None,
    allow_empty_classes: bool = False,
) -> dict[str, Any]:
    class_root = find_class_root(descriptor)
    class_dirs = discover_class_dirs(class_root, descriptor.ignore_hidden_files)
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]
    if not class_dirs:
        raise ValueError(f"No class folders found under {class_root}")

    class_to_idx = build_class_to_idx(class_dirs, descriptor.class_name_mapping)
    suffixes = set(descriptor.image_extensions)
    samples: list[DatasetSample] = []
    class_summary: list[dict[str, Any]] = []
    unsupported_extensions: Counter[str] = Counter()
    critical_errors: list[str] = []

    for class_dir in class_dirs:
        class_name = mapped_class_name(class_dir.name, descriptor.class_name_mapping)
        supported_files, unsupported = list_supported_files(
            class_dir,
            suffixes,
            ignore_hidden_files=descriptor.ignore_hidden_files,
            follow_symlinks=descriptor.follow_symlinks,
        )
        unsupported_extensions.update(unsupported)
        total_supported = len(supported_files)
        if max_samples_per_class is not None:
            supported_files = supported_files[:max_samples_per_class]
        if total_supported == 0:
            message = f"Class folder contains no supported images: {class_dir}"
            if not allow_empty_classes:
                critical_errors.append(message)
        if total_supported < descriptor.min_images_per_class:
            critical_errors.append(
                f"Class {class_name} has {total_supported} supported images, fewer than min_images_per_class={descriptor.min_images_per_class}"
            )
        label = class_to_idx[class_name]
        for image_path in supported_files:
            samples.append(DatasetSample(path=str(image_path), label=label, class_name=class_name))
        class_summary.append(
            {
                "class_name": class_name,
                "class_idx": label,
                "class_dir": str(class_dir),
                "num_supported_images": total_supported,
                "num_used_images": len(supported_files),
                "is_empty": total_supported == 0,
            }
        )

    if descriptor.expected_num_classes is not None and len(class_dirs) != descriptor.expected_num_classes:
        critical_errors.append(
            f"Expected {descriptor.expected_num_classes} classes, found {len(class_dirs)} under {class_root}"
        )

    return {
        "dataset": descriptor.name,
        "display_name": descriptor.display_name,
        "dataset_root": str(descriptor.root),
        "class_root": str(class_root),
        "is_valid": not critical_errors,
        "critical_errors": critical_errors,
        "num_classes": len(class_dirs),
        "num_samples": len(samples),
        "class_to_idx": class_to_idx,
        "class_summary": sorted(class_summary, key=lambda row: row["class_idx"]),
        "unsupported_extensions": dict(sorted(unsupported_extensions.items())),
        "image_extensions": list(descriptor.image_extensions),
        "ignore_hidden_files": descriptor.ignore_hidden_files,
        "follow_symlinks": descriptor.follow_symlinks,
        "samples": samples,
    }


def list_class_folder_samples(
    descriptor: DatasetDescriptor,
    max_samples_per_class: int | None = None,
    min_samples_per_class: int | None = None,
    allow_empty_classes: bool = False,
) -> tuple[list[DatasetSample], dict[str, int]]:
    if min_samples_per_class is not None:
        descriptor = replace(descriptor, min_images_per_class=min_samples_per_class)
    report = inspect_class_folder_dataset(
        descriptor,
        max_samples_per_class=max_samples_per_class,
        allow_empty_classes=allow_empty_classes,
    )
    if not report["is_valid"]:
        raise ValueError("; ".join(report["critical_errors"]))
    return list(report["samples"]), dict(report["class_to_idx"])


def list_supported_files(
    class_dir: Path,
    suffixes: set[str],
    ignore_hidden_files: bool,
    follow_symlinks: bool,
) -> tuple[list[Path], Counter[str]]:
    supported: list[Path] = []
    unsupported: Counter[str] = Counter()
    for path in class_dir.rglob("*"):
        if ignore_hidden_files and any(part.startswith(".") for part in path.relative_to(class_dir).parts):
            continue
        if path.is_symlink() and not follow_symlinks:
            continue
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in suffixes:
            supported.append(path)
        else:
            unsupported[suffix or "<no_extension>"] += 1
    return sorted(supported, key=lambda path: str(path).lower()), unsupported


def build_class_to_idx(class_dirs: list[Path], class_name_mapping: dict[str, str] | None) -> dict[str, int]:
    names = [mapped_class_name(path.name, class_name_mapping) for path in class_dirs]
    if len(names) != len(set(names)):
        raise ValueError("Class name mapping produced duplicate class names")
    return {name: index for index, name in enumerate(names)}


def mapped_class_name(raw_name: str, class_name_mapping: dict[str, str] | None) -> str:
    if not class_name_mapping:
        return raw_name
    return class_name_mapping.get(raw_name, raw_name)


def normalize_extensions(extensions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = []
    for extension in extensions:
        value = str(extension).lower()
        normalized.append(value if value.startswith(".") else f".{value}")
    return tuple(sorted(set(normalized)))


def is_hidden(path: Path) -> bool:
    return path.name.startswith(".")
