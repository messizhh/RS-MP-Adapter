from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Iterable

from src.datasets.base_dataset import DatasetDescriptor, DatasetSample, find_class_root, list_class_folder_samples
from src.datasets.dataset_registry import get_dataset_descriptor
from src.utils.timing import utc_now_iso


DEFAULT_SHOTS = (1, 2, 4, 8, 16)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


def sample_to_dict(sample: DatasetSample, path_root: str | Path | None = None) -> dict[str, object]:
    return {"path": portable_sample_path(sample.path, path_root), "label": sample.label, "class_name": sample.class_name}


def portable_sample_path(path: str, path_root: str | Path | None = None) -> str:
    if path_root is None:
        return path
    try:
        return Path(path).resolve().relative_to(Path(path_root).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"sample path is not under path_root: sample={path}, path_root={path_root}") from exc


def split_samples_by_class(
    samples: Iterable[DatasetSample],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[list[DatasetSample], list[DatasetSample], list[DatasetSample]]:
    rng = Random(seed)
    by_class: dict[int, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.label].append(sample)

    train: list[DatasetSample] = []
    val: list[DatasetSample] = []
    test: list[DatasetSample] = []
    for label in sorted(by_class):
        shuffled = sorted(by_class[label], key=lambda item: item.path)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio)) if n_total >= 3 else 0
        if n_train + n_val >= n_total and n_total > 1:
            n_val = max(0, n_total - n_train - 1)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])
    return sorted(train, key=lambda item: item.path), sorted(val, key=lambda item: item.path), sorted(test, key=lambda item: item.path)


def build_support_set(train_samples: Iterable[DatasetSample], shot: int, seed: int) -> list[DatasetSample]:
    rng = Random(seed)
    by_class: dict[int, list[DatasetSample]] = defaultdict(list)
    for sample in train_samples:
        by_class[sample.label].append(sample)
    support: list[DatasetSample] = []
    for label in sorted(by_class):
        candidates = sorted(by_class[label], key=lambda item: item.path)
        if len(candidates) < shot:
            class_name = candidates[0].class_name if candidates else str(label)
            raise ValueError(
                f"Class {class_name} (label={label}) has {len(candidates)} train samples, fewer than shot={shot}"
            )
        rng.shuffle(candidates)
        support.extend(candidates[:shot])
    return sorted(support, key=lambda item: item.path)


def make_split_payload(
    dataset: str,
    seed: int,
    shot: int | None,
    train: list[DatasetSample],
    val: list[DatasetSample],
    test: list[DatasetSample],
    support: list[DatasetSample],
    class_to_idx: dict[str, int],
    source_script: str,
    dataset_root: str = "",
    split_policy: str = "class_stratified_random",
    split_ratios: dict[str, float] | None = None,
    image_extensions: list[str] | None = None,
    execution_env: str = "local_wsl",
    run_mode: str = "smoke_test",
    is_paper_result: bool = False,
    sample_path_root: str | Path | None = None,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": seed,
        "shot": shot,
        "train": [sample_to_dict(sample, sample_path_root) for sample in train],
        "val": [sample_to_dict(sample, sample_path_root) for sample in val],
        "test": [sample_to_dict(sample, sample_path_root) for sample in test],
        "support": [sample_to_dict(sample, sample_path_root) for sample in support],
        "class_to_idx": class_to_idx,
        "created_at": utc_now_iso(),
        "source_script": source_script,
        "dataset_root": dataset_root,
        "split_policy": split_policy,
        "split_ratios": split_ratios or {"train": 0.6, "val": 0.2, "test": 0.2},
        "image_extensions": image_extensions or [],
        "num_classes": len(class_to_idx),
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "num_support": len(support),
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": is_paper_result,
    }


def write_split_json(path: str | Path, data: dict[str, object], overwrite: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temp_name, destination)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return destination


def generate_split_files(
    dataset: str,
    root: str | Path,
    output_dir: str | Path,
    shots: Iterable[int] = DEFAULT_SHOTS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    source_script: str = "scripts/generate_splits.py",
    overwrite: bool = False,
    dry_run: bool = False,
    max_samples_per_class: int | None = None,
    min_samples_per_class: int | None = None,
    descriptor: DatasetDescriptor | None = None,
    split_policy: str = "class_stratified_random",
    execution_env: str = "local_wsl",
    run_mode: str = "smoke_test",
    is_paper_result: bool = False,
) -> list[Path]:
    descriptor = descriptor or get_dataset_descriptor(dataset, root=root)
    samples, class_to_idx = list_class_folder_samples(
        descriptor,
        max_samples_per_class=max_samples_per_class,
        min_samples_per_class=min_samples_per_class,
    )
    class_root = find_class_root(descriptor)
    output = Path(output_dir)
    written: list[Path] = []
    split_ratios = {"train": train_ratio, "val": val_ratio, "test": max(0.0, 1.0 - train_ratio - val_ratio)}

    for seed in seeds:
        train, val, test = split_samples_by_class(samples, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
        base_payload = make_split_payload(
            dataset,
            seed,
            None,
            train,
            val,
            test,
            [],
            class_to_idx,
            source_script,
            dataset_root="",
            split_policy=split_policy,
            split_ratios=split_ratios,
            image_extensions=list(descriptor.image_extensions),
            execution_env=execution_env,
            run_mode=run_mode,
            is_paper_result=is_paper_result,
            sample_path_root=class_root,
        )
        base_path = output / f"base_split_seed{seed}.json"
        if not dry_run:
            written.append(write_split_json(base_path, base_payload, overwrite=overwrite))
        else:
            written.append(base_path)
        for shot in shots:
            support = build_support_set(train, shot=shot, seed=seed)
            payload = make_split_payload(
                dataset,
                seed,
                shot,
                train,
                val,
                test,
                support,
                class_to_idx,
                source_script,
                dataset_root="",
                split_policy=split_policy,
                split_ratios=split_ratios,
                image_extensions=list(descriptor.image_extensions),
                execution_env=execution_env,
                run_mode=run_mode,
                is_paper_result=is_paper_result,
                sample_path_root=class_root,
            )
            path = output / f"shot_{shot}_seed{seed}.json"
            if not dry_run:
                written.append(write_split_json(path, payload, overwrite=overwrite))
            else:
                written.append(path)
    return written
