from __future__ import annotations

from src.baselines.method_base import MethodBase
from src.prototypes.cache_compressor import CacheCompressor
from src.prototypes.prototype_logits import PrototypeLogits
from src.utils.features import add_matrices, cosine_similarity_matrix, to_labels, to_rows


class RsCpcAdapter(MethodBase):
    method_name = "rs_cpc"

    def __init__(
        self,
        num_prototypes_per_class: int = 2,
        prototype_init: str = "random_group_mean",
        seed: int = 1,
        temperature: float = 1.0,
        alpha: float = 1.0,
        text_features=None,
    ) -> None:
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_init = prototype_init
        self.seed = seed
        self.temperature = temperature
        self.alpha = alpha
        self.text_features = to_rows(text_features) if text_features is not None else None
        self.prototypes: list[list[float]] = []
        self.prototype_labels: list[int] = []
        self.num_classes = 0
        self.compression_info: dict[str, object] = {}
        self.class_names: list[str] | None = None

    def fit(self, support_features, support_labels, val_features=None, val_labels=None):
        labels = to_labels(support_labels)
        self.num_classes = max(labels) + 1
        compressed = CacheCompressor(self.num_prototypes_per_class, self.prototype_init, self.seed).compress(support_features, labels)
        self.prototypes = compressed["prototypes"]
        self.prototype_labels = compressed["prototype_labels"]
        self.compression_info = compressed
        return self

    def predict_logits(self, image_features):
        if not self.prototypes:
            raise ValueError("RsCpcAdapter must be fit before predict_logits")
        logits = PrototypeLogits(self.temperature).compute(image_features, self.prototypes, self.prototype_labels, self.num_classes)
        logits = [[self.alpha * score for score in row] for row in logits]
        if self.text_features is None:
            return logits
        return add_matrices(cosine_similarity_matrix(image_features, self.text_features), logits)

    @property
    def cache_entries(self) -> int:
        return len(self.prototypes)

    @property
    def compression_ratio(self) -> float:
        return float(self.compression_info.get("compression_ratio", 0.0))
