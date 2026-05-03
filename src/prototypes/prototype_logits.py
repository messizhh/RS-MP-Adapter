from __future__ import annotations

from src.utils.features import cosine_similarity_matrix, one_hot, matmul


class PrototypeLogits:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def compute(self, image_features, prototypes, prototype_labels, num_classes: int):
        sims = cosine_similarity_matrix(image_features, prototypes)
        values = one_hot(prototype_labels, num_classes)
        logits = matmul(sims, values)
        return [[score / self.temperature for score in row] for row in logits]
