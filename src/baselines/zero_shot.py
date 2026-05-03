from __future__ import annotations

import time
from dataclasses import dataclass

from src.features.feature_cache import FeatureCache


@dataclass(frozen=True)
class ZeroShotResult:
    top1_acc: float
    num_samples: int
    num_classes: int
    inference_time_sec: float
    images_per_second: float
    used_fake_features: bool


class ZeroShotClassifier:
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def predict_logits(self, cache: FeatureCache) -> torch.Tensor:
        cache.validate()
        if cache.text_features is None:
            raise ValueError("Zero-shot evaluation requires text_features in the feature cache")
        image_features = to_rows(cache.image_features)
        text_features = to_rows(cache.text_features)
        return [
            [dot(image_row, text_row) / self.temperature for text_row in text_features]
            for image_row in image_features
        ]

    def evaluate(self, cache: FeatureCache) -> ZeroShotResult:
        start = time.perf_counter()
        logits = self.predict_logits(cache)
        labels = to_labels(cache.labels)
        predictions = [max(range(len(row)), key=row.__getitem__) for row in logits]
        correct = sum(int(prediction == label) for prediction, label in zip(predictions, labels)) / len(labels)
        elapsed = time.perf_counter() - start
        num_samples = len(labels)
        return ZeroShotResult(
            top1_acc=float(correct),
            num_samples=num_samples,
            num_classes=len(cache.class_names),
            inference_time_sec=elapsed,
            images_per_second=float(num_samples / elapsed) if elapsed > 0 else 0.0,
            used_fake_features=not bool(cache.metadata.get("is_real_feature_extraction", False)),
        )


def to_rows(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return [list(row) for row in value]


def to_labels(value):
    if hasattr(value, "detach"):
        return [int(item) for item in value.detach().cpu().tolist()]
    if hasattr(value, "tolist"):
        return [int(item) for item in value.tolist()]
    return [int(item) for item in value]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
