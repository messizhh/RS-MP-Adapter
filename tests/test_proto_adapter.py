from __future__ import annotations

import unittest

from src.baselines.proto_adapter import ProtoAdapter
from src.baselines.proto_adapter_f import ProtoAdapterF
from src.baselines.runner_utils import split_support_query
from src.features.feature_cache import make_fake_feature_cache


class ProtoAdapterTest(unittest.TestCase):
    def test_proto_adapter_cache_entries(self) -> None:
        cache = make_fake_feature_cache(num_samples=12, num_classes=3, feature_dim=8)
        split = split_support_query(cache, shot=2)
        method = ProtoAdapter().fit(split["support_features"], split["support_labels"])
        self.assertEqual(method.cache_entries, 3)
        logits = method.predict_logits(split["query_features"])
        self.assertEqual(len(logits[0]), 3)

    def test_proto_adapter_f_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ProtoAdapterF().fit([], [])


if __name__ == "__main__":
    unittest.main()
