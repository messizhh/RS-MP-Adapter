from __future__ import annotations

import torch


def per_class_accuracy(predictions: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum().item() == 0:
            result[class_name] = 0.0
        else:
            result[class_name] = float((predictions[mask] == labels[mask]).float().mean().item())
    return result
