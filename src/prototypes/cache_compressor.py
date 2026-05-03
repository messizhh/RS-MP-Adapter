from __future__ import annotations

from src.prototypes.prototype_builder import PrototypeBuilder


class CacheCompressor:
    def __init__(self, num_prototypes_per_class: int = 1, init_mode: str = "mean", seed: int = 1) -> None:
        self.builder = PrototypeBuilder(num_prototypes_per_class, init_mode, seed)

    def compress(self, support_features, support_labels) -> dict[str, object]:
        prototypes, prototype_labels, info = self.builder.build(support_features, support_labels)
        original = len(support_labels)
        compressed = len(prototypes)
        return {
            "prototypes": prototypes,
            "prototype_labels": prototype_labels,
            "assignments": info["assignments"],
            "original_cache_entries": original,
            "compressed_cache_entries": compressed,
            "compression_ratio": (original / compressed) if compressed else 0.0,
            "num_prototypes_per_class": self.builder.num_prototypes_per_class,
            "init_mode": self.builder.init_mode,
        }
