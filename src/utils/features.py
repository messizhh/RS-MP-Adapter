from __future__ import annotations

import json
import math
from typing import Any


def to_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [[float(item) for item in row] for row in value]


def to_labels(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def normalize_rows(rows: Any) -> list[list[float]]:
    return [l2_normalize(row) for row in to_rows(rows)]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def cosine_similarity_matrix(left: Any, right: Any) -> list[list[float]]:
    left_rows = normalize_rows(left)
    right_rows = normalize_rows(right)
    return [[dot(left_row, right_row) for right_row in right_rows] for left_row in left_rows]


def one_hot(labels: Any, num_classes: int) -> list[list[float]]:
    rows = []
    for label in to_labels(labels):
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label {label} is outside num_classes={num_classes}")
        row = [0.0] * num_classes
        row[label] = 1.0
        rows.append(row)
    return rows


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    columns = list(zip(*right))
    return [[sum(a * b for a, b in zip(row, column)) for column in columns] for row in left]


def add_matrices(left: list[list[float]], right: list[list[float]], left_weight: float = 1.0, right_weight: float = 1.0) -> list[list[float]]:
    if len(left) != len(right):
        raise ValueError("Matrix row counts must match")
    return [
        [left_weight * a + right_weight * b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def argmax_rows(logits: list[list[float]]) -> list[int]:
    return [max(range(len(row)), key=row.__getitem__) for row in logits]


def count_cache_entries(features: Any) -> int:
    return len(to_rows(features))


def count_trainable_params(obj: Any = None) -> int:
    return 0


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value))
