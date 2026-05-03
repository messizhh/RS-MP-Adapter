from __future__ import annotations

from typing import Any

from src.eval.metrics import argmax_rows, confusion_matrix, per_class_accuracy, to_labels, to_rows, top1_accuracy


def evaluate_logits(logits: Any, labels: Any, class_names: list[str] | None = None) -> dict[str, Any]:
    rows = to_rows(logits)
    label_list = to_labels(labels)
    if len(rows) != len(label_list):
        raise ValueError("logits and labels must have matching first dimensions")
    if class_names is None:
        num_classes = len(rows[0]) if rows else 0
        class_names = [str(index) for index in range(num_classes)]
    else:
        num_classes = len(class_names)
    predictions = argmax_rows(rows)
    return {
        "top1_acc": top1_accuracy(rows, label_list),
        "per_class_acc": per_class_accuracy(predictions, label_list, class_names=class_names, num_classes=num_classes),
        "confusion_matrix": confusion_matrix(predictions, label_list, num_classes),
        "num_samples": len(label_list),
        "num_classes": num_classes,
        "predictions": predictions,
    }
