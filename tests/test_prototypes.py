from __future__ import annotations

import unittest

from src.prototypes.cache_compressor import CacheCompressor
from src.prototypes.prototype_builder import PrototypeBuilder


class PrototypeTest(unittest.TestCase):
    def test_random_group_mean_deterministic(self) -> None:
        features = [[float(i), 1.0] for i in range(12)]
        labels = [0] * 4 + [1] * 4 + [2] * 4
        first = PrototypeBuilder(2, "random_group_mean", seed=7).build(features, labels)
        second = PrototypeBuilder(2, "random_group_mean", seed=7).build(features, labels)
        self.assertEqual(first, second)

    def test_cache_compression_ratio(self) -> None:
        features = [[float(i), 1.0] for i in range(12)]
        labels = [0] * 4 + [1] * 4 + [2] * 4
        compressed = CacheCompressor(2, "random_group_mean", seed=1).compress(features, labels)
        self.assertEqual(compressed["compressed_cache_entries"], 6)
        self.assertEqual(compressed["original_cache_entries"], 12)
        self.assertEqual(compressed["compression_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
