from __future__ import annotations

import torch


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(label), int(prediction)] += 1
    return matrix
