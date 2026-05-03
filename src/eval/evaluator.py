from __future__ import annotations

import torch

from src.eval.confusion_matrix import confusion_matrix
from src.eval.metrics import top1_accuracy
from src.eval.per_class_accuracy import per_class_accuracy


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> dict[str, object]:
    predictions = logits.argmax(dim=1)
    return {
        "top1_acc": top1_accuracy(logits, labels),
        "per_class_acc": per_class_accuracy(predictions, labels, class_names),
        "confusion_matrix": confusion_matrix(predictions, labels, len(class_names)).tolist(),
    }
