from __future__ import annotations

from src.prototypes.prototype_builder import PrototypeBuilder


class PrototypeSelector:
    def __init__(self, mode: str = "random_group_mean", seed: int = 1) -> None:
        self.mode = mode
        self.seed = seed

    def select(self, features, labels, num_prototypes_per_class: int):
        builder = PrototypeBuilder(num_prototypes_per_class, self.mode, self.seed)
        return builder.build(features, labels)
