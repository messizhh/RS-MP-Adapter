from __future__ import annotations

from collections import defaultdict
from random import Random

from src.utils.features import cosine_similarity_matrix, l2_normalize, to_labels, to_rows


class PrototypeBuilder:
    def __init__(self, num_prototypes_per_class: int = 1, init_mode: str = "mean", seed: int = 1) -> None:
        if num_prototypes_per_class <= 0:
            raise ValueError("num_prototypes_per_class must be positive")
        self.num_prototypes_per_class = num_prototypes_per_class
        self.init_mode = init_mode
        self.seed = seed

    def build(self, features, labels) -> tuple[list[list[float]], list[int], dict[str, object]]:
        grouped: dict[int, list[list[float]]] = defaultdict(list)
        for row, label in zip(to_rows(features), to_labels(labels)):
            grouped[label].append(row)
        prototypes: list[list[float]] = []
        prototype_labels: list[int] = []
        assignments: dict[int, list[list[int]]] = {}
        for label in sorted(grouped):
            class_prototypes, class_assignments = self._build_for_class(grouped[label], label)
            prototypes.extend(class_prototypes)
            prototype_labels.extend([label] * len(class_prototypes))
            assignments[label] = class_assignments
        return prototypes, prototype_labels, {"assignments": assignments, "init_mode": self.init_mode}

    def _build_for_class(self, rows: list[list[float]], label: int) -> tuple[list[list[float]], list[list[int]]]:
        m = min(self.num_prototypes_per_class, len(rows))
        if self.init_mode == "mean":
            prototype = mean_feature(rows)
            return [prototype for _ in range(m)], [list(range(len(rows))) for _ in range(m)]
        if self.init_mode == "random_group_mean":
            rng = Random(self.seed + label)
            indices = list(range(len(rows)))
            rng.shuffle(indices)
            groups = [indices[index::m] for index in range(m)]
            return [mean_feature([rows[idx] for idx in group]) for group in groups], groups
        if self.init_mode == "medoid":
            selected = select_diverse_medoids(rows, m)
            return [rows[idx] for idx in selected], [[idx] for idx in selected]
        if self.init_mode == "kmeans":
            # Dependency-free Phase 1E fallback: deterministic random groups.
            fallback = PrototypeBuilder(m, "random_group_mean", self.seed)
            return fallback._build_for_class(rows, label)
        raise ValueError(f"Unsupported prototype init mode: {self.init_mode}")


def mean_feature(rows: list[list[float]]) -> list[float]:
    if not rows:
        raise ValueError("Cannot build a prototype from an empty group")
    return l2_normalize([sum(values) / len(rows) for values in zip(*rows)])


def select_diverse_medoids(rows: list[list[float]], count: int) -> list[int]:
    if count <= 0:
        return []
    class_mean = mean_feature(rows)
    mean_sims = cosine_similarity_matrix(rows, [class_mean])
    selected = [min(range(len(rows)), key=lambda idx: (-mean_sims[idx][0], idx))]
    sims = cosine_similarity_matrix(rows, rows)
    while len(selected) < min(count, len(rows)):
        candidates = [idx for idx in range(len(rows)) if idx not in selected]
        next_idx = max(
            candidates,
            key=lambda idx: (1.0 - max(sims[idx][selected_idx] for selected_idx in selected), -idx),
        )
        selected.append(next_idx)
    return selected
