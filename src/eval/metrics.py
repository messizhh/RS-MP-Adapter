from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def to_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    rows = [list(row) for row in value]
    if not rows or not all(isinstance(row, list) for row in rows):
        raise ValueError("Expected a 2D logits/features-like value")
    return rows


def to_labels(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    labels = [int(item) for item in value]
    if not labels:
        raise ValueError("Expected at least one label")
    return labels


def argmax_rows(logits: Any) -> list[int]:
    rows = to_rows(logits)
    return [max(range(len(row)), key=row.__getitem__) for row in rows]


def top1_accuracy(logits: Any, labels: Any) -> float:
    predictions = argmax_rows(logits)
    label_list = to_labels(labels)
    if len(predictions) != len(label_list):
        raise ValueError("logits and labels must have matching first dimensions")
    return sum(int(pred == label) for pred, label in zip(predictions, label_list)) / len(label_list)


def confusion_matrix(predictions: Any, labels: Any, num_classes: int) -> list[list[int]]:
    pred_list = to_labels(predictions)
    label_list = to_labels(labels)
    if len(pred_list) != len(label_list):
        raise ValueError("predictions and labels must have matching length")
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, prediction in zip(label_list, pred_list):
        if label < 0 or label >= num_classes or prediction < 0 or prediction >= num_classes:
            raise ValueError("label or prediction is outside num_classes")
        matrix[label][prediction] += 1
    return matrix


def per_class_accuracy(
    predictions: Any,
    labels: Any,
    class_names: list[str] | None = None,
    num_classes: int | None = None,
) -> list[dict[str, Any]]:
    pred_list = to_labels(predictions)
    label_list = to_labels(labels)
    if len(pred_list) != len(label_list):
        raise ValueError("predictions and labels must have matching length")
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = max(label_list) + 1
    names = class_names or [str(index) for index in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError("class_names length must match num_classes")

    counts: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for label, prediction in zip(label_list, pred_list):
        counts[label] += 1
        correct[label] += int(label == prediction)

    rows: list[dict[str, Any]] = []
    for class_idx in range(num_classes):
        num_samples = counts[class_idx]
        num_correct = correct[class_idx]
        rows.append(
            {
                "class_name": names[class_idx],
                "class_idx": class_idx,
                "num_samples": num_samples,
                "num_correct": num_correct,
                "accuracy": (num_correct / num_samples) if num_samples else 0.0,
            }
        )
    return rows


def mean_std_over_seeds(values: Iterable[float]) -> dict[str, Any]:
    value_list = [float(value) for value in values]
    if not value_list:
        return {"mean": None, "std": None, "num_seeds": 0}
    mean = sum(value_list) / len(value_list)
    if len(value_list) == 1:
        std = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in value_list) / (len(value_list) - 1)
        std = math.sqrt(variance)
    return {"mean": mean, "std": std, "num_seeds": len(value_list)}
