from __future__ import annotations

from collections import defaultdict

from src.baselines.method_base import MethodBase
from src.utils.features import cosine_similarity_matrix, l2_normalize, to_labels, to_rows


class LinearProbe(MethodBase):
    """CPU-safe nearest-centroid skeleton for Phase 1E dry-run validation."""

    method_name = "linear_probe"

    def __init__(self) -> None:
        self.prototypes: list[list[float]] = []
        self.class_names: list[str] | None = None
        self.trainable_params = 0

    def fit(self, support_features, support_labels, val_features=None, val_labels=None):
        rows = to_rows(support_features)
        labels = to_labels(support_labels)
        grouped: dict[int, list[list[float]]] = defaultdict(list)
        for row, label in zip(rows, labels):
            grouped[label].append(row)
        self.prototypes = []
        for label in sorted(grouped):
            count = len(grouped[label])
            mean = [sum(values) / count for values in zip(*grouped[label])]
            self.prototypes.append(l2_normalize(mean))
        self.trainable_params = len(self.prototypes) * (len(self.prototypes[0]) if self.prototypes else 0)
        return self

    def predict_logits(self, image_features):
        if not self.prototypes:
            raise ValueError("LinearProbe must be fit before predict_logits")
        return cosine_similarity_matrix(image_features, self.prototypes)
