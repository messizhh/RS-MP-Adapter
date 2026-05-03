from __future__ import annotations

import torch


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_samples, num_classes]")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [num_samples]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have matching first dimensions")
    return float((logits.argmax(dim=1) == labels).float().mean().item())
