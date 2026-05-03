from __future__ import annotations

import unittest

from src.baselines.runner_utils import split_support_query
from src.baselines.tip_adapter import TipAdapter
from src.baselines.tip_adapter_f import TipAdapterF
from src.features.feature_cache import make_fake_feature_cache


class TipAdapterTest(unittest.TestCase):
    def test_tip_adapter_cache_entries(self) -> None:
        cache = make_fake_feature_cache(num_samples=12, num_classes=3, feature_dim=8)
        split = split_support_query(cache, shot=2)
        method = TipAdapter().fit(split["support_features"], split["support_labels"])
        self.assertEqual(method.cache_entries, 6)
        logits = method.predict_logits(split["query_features"])
        self.assertEqual(len(logits[0]), 3)

    def test_tip_adapter_f_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            TipAdapterF().fit([], [])


if __name__ == "__main__":
    unittest.main()
