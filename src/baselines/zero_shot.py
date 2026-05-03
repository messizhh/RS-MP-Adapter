from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.baselines.method_base import MethodBase
from src.eval.evaluator import evaluate_logits
from src.features.feature_cache import FeatureCache
from src.utils.features import cosine_similarity_matrix, to_rows


@dataclass(frozen=True)
class ZeroShotResult:
    logits: list[list[float]]
    predictions: list[int]
    metrics: dict[str, Any]
    inference_time_sec: float
    images_per_second: float
    used_fake_features: bool


class ZeroShotClassifier(MethodBase):
    method_name = "zero_shot_clip"

    def __init__(self, temperature: float = 1.0, similarity: str = "cosine", normalize_features: bool = True) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if similarity != "cosine":
            raise ValueError("Only cosine similarity is supported")
        self.temperature = temperature
        self.similarity = similarity
        self.normalize_features = normalize_features
        self.text_features: list[list[float]] | None = None
        self.class_names: list[str] | None = None

    def fit(self, support_features, support_labels=None, val_features=None, val_labels=None):
        self.text_features = to_rows(support_features)
        return self

    def predict_logits(self, image_features) -> list[list[float]]:
        if self.text_features is None:
            raise ValueError("ZeroShotClassifier requires text_features via fit() or evaluate(cache)")
        logits = cosine_similarity_matrix(image_features, self.text_features)
        return [[score / self.temperature for score in row] for row in logits]

    def evaluate(self, image_features, labels=None):
        if isinstance(image_features, FeatureCache):
            return self.evaluate_cache(image_features)
        return evaluate_logits(self.predict_logits(image_features), labels, class_names=self.class_names)

    def evaluate_cache(self, cache: FeatureCache) -> ZeroShotResult:
        cache.validate()
        if cache.text_features is None:
            raise ValueError("Zero-shot evaluation requires text_features in the feature cache")
        self.fit(cache.text_features)
        self.class_names = cache.class_names
        start = time.perf_counter()
        logits = self.predict_logits(cache.image_features)
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
