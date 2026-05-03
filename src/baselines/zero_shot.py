from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from src.eval.evaluator import evaluate_logits
from src.features.feature_cache import FeatureCache


@dataclass(frozen=True)
class ZeroShotResult:
    logits: list[list[float]]
    predictions: list[int]
    metrics: dict[str, Any]
    inference_time_sec: float
    images_per_second: float
    used_fake_features: bool


class ZeroShotClassifier:
    def __init__(
        self,
        temperature: float = 1.0,
        similarity: str = "cosine",
        normalize_features: bool = True,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if similarity != "cosine":
            raise ValueError("Only cosine similarity is supported in Phase 1B")
        self.temperature = temperature
        self.similarity = similarity
        self.normalize_features = normalize_features

    def predict_logits(self, cache: FeatureCache) -> list[list[float]]:
        cache.validate()
        if cache.text_features is None:
            raise ValueError("Zero-shot evaluation requires text_features in the feature cache")
        image_features = to_rows(cache.image_features)
        text_features = to_rows(cache.text_features)
        if self.normalize_features:
            image_features = [l2_normalize(row) for row in image_features]
            text_features = [l2_normalize(row) for row in text_features]
        return [
            [dot(image_row, text_row) / self.temperature for text_row in text_features]
            for image_row in image_features
        ]

    def evaluate(self, cache: FeatureCache) -> ZeroShotResult:
        start = time.perf_counter()
        logits = self.predict_logits(cache)
        metrics = evaluate_logits(logits, cache.image_labels, class_names=cache.class_names)
        elapsed = time.perf_counter() - start
        num_samples = int(metrics["num_samples"])
        return ZeroShotResult(
            logits=logits,
            predictions=list(metrics["predictions"]),
            metrics=metrics,
            inference_time_sec=elapsed,
            images_per_second=float(num_samples / elapsed) if elapsed > 0 else 0.0,
            used_fake_features=not bool(cache.metadata.get("is_real_feature_extraction", False)),
        )


def to_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [[float(item) for item in row] for row in value]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]
