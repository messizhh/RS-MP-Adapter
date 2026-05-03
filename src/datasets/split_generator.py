from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Iterable

from src.datasets.base_dataset import DatasetSample, list_class_folder_samples
from src.datasets.dataset_registry import get_dataset_descriptor
from src.utils.io import write_json
from src.utils.timing import utc_now_iso


DEFAULT_SHOTS = (1, 2, 4, 8, 16)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


def sample_to_dict(sample: DatasetSample) -> dict[str, object]:
    return {"path": sample.path, "label": sample.label, "class_name": sample.class_name}


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
    for class_samples in by_class.values():
        shuffled = list(class_samples)
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
        candidates = list(by_class[label])
        if len(candidates) < shot:
            raise ValueError(f"Class {label} has {len(candidates)} train samples, fewer than shot={shot}")
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
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": seed,
        "shot": shot,
        "train": [sample_to_dict(sample) for sample in train],
        "val": [sample_to_dict(sample) for sample in val],
        "test": [sample_to_dict(sample) for sample in test],
        "support": [sample_to_dict(sample) for sample in support],
        "class_to_idx": class_to_idx,
        "created_at": utc_now_iso(),
        "source_script": source_script,
    }


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
) -> list[Path]:
    descriptor = get_dataset_descriptor(dataset, root=root)
    samples, class_to_idx = list_class_folder_samples(descriptor)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for seed in seeds:
        train, val, test = split_samples_by_class(samples, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
        base_payload = make_split_payload(dataset, seed, None, train, val, test, [], class_to_idx, source_script)
        written.append(write_json(output / f"base_split_seed{seed}.json", base_payload, overwrite=overwrite))
        for shot in shots:
            support = build_support_set(train, shot=shot, seed=seed)
            payload = make_split_payload(dataset, seed, shot, train, val, test, support, class_to_idx, source_script)
            written.append(write_json(output / f"shot_{shot}_seed{seed}.json", payload, overwrite=overwrite))
    return written
