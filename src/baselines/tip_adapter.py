from __future__ import annotations

import math

from src.baselines.method_base import MethodBase
from src.utils.features import add_matrices, cosine_similarity_matrix, matmul, one_hot, to_labels, to_rows


class TipAdapter(MethodBase):
    method_name = "tip_adapter"

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, temperature: float = 1.0, text_features=None) -> None:
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.text_features = to_rows(text_features) if text_features is not None else None
        self.cache_keys: list[list[float]] = []
        self.cache_values: list[list[float]] = []
        self.num_classes = 0
        self.class_names: list[str] | None = None

    def fit(self, support_features, support_labels, val_features=None, val_labels=None):
        labels = to_labels(support_labels)
        self.num_classes = max(labels) + 1
        self.cache_keys = to_rows(support_features)
        self.cache_values = one_hot(labels, self.num_classes)
        return self

    def predict_logits(self, image_features):
        if not self.cache_keys:
            raise ValueError("TipAdapter must be fit before predict_logits")
        affinity = cosine_similarity_matrix(image_features, self.cache_keys)
        cache_weights = [[math.exp(self.beta * (score - 1.0)) for score in row] for row in affinity]
        cache_logits = matmul(cache_weights, self.cache_values)
        cache_logits = [[score * self.alpha / self.temperature for score in row] for row in cache_logits]
        if self.text_features is None:
            return cache_logits
        zero_shot_logits = cosine_similarity_matrix(image_features, self.text_features)
        return add_matrices(zero_shot_logits, cache_logits)

    @property
    def cache_entries(self) -> int:
        return len(self.cache_keys)
