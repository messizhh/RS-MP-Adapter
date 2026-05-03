from __future__ import annotations

from collections import defaultdict

from src.baselines.method_base import MethodBase
from src.utils.features import add_matrices, cosine_similarity_matrix, l2_normalize, to_labels, to_rows


class ProtoAdapter(MethodBase):
    method_name = "proto_adapter"

    def __init__(self, alpha: float = 1.0, temperature: float = 1.0, text_features=None) -> None:
        self.alpha = alpha
        self.temperature = temperature
        self.text_features = to_rows(text_features) if text_features is not None else None
        self.prototypes: list[list[float]] = []
        self.class_names: list[str] | None = None

    def fit(self, support_features, support_labels, val_features=None, val_labels=None):
        grouped: dict[int, list[list[float]]] = defaultdict(list)
        for row, label in zip(to_rows(support_features), to_labels(support_labels)):
            grouped[label].append(row)
        self.prototypes = []
        for label in sorted(grouped):
            count = len(grouped[label])
            mean = [sum(values) / count for values in zip(*grouped[label])]
            self.prototypes.append(l2_normalize(mean))
        return self

    def predict_logits(self, image_features):
        if not self.prototypes:
            raise ValueError("ProtoAdapter must be fit before predict_logits")
        proto_logits = cosine_similarity_matrix(image_features, self.prototypes)
        proto_logits = [[self.alpha * score / self.temperature for score in row] for row in proto_logits]
        if self.text_features is None:
            return proto_logits
        return add_matrices(cosine_similarity_matrix(image_features, self.text_features), proto_logits)

    @property
    def cache_entries(self) -> int:
        return len(self.prototypes)
