from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.models.base_backbone import BackboneUnavailableError, create_backbone, expand_prompts


class BackboneTest(unittest.TestCase):
    def test_fake_backbone_shapes(self) -> None:
        backbone = create_backbone("fake_backbone", {"backbone": {"name": "fake_backbone", "family": "fake", "feature_dim": 8}}, dry_run=True, device="cpu")
        backbone.load_model().eval()
        image_features = backbone.encode_images(["a.jpg", "b.jpg"])
        text_features = backbone.encode_text(["a satellite photo of forest.", "a satellite photo of river."])
        self.assertEqual(len(image_features), 2)
        self.assertEqual(len(image_features[0]), 8)
        self.assertEqual(len(text_features), 2)
        self.assertEqual(len(text_features[0]), 8)
        self.assertEqual(backbone.get_feature_dim(), 8)
        self.assertTrue(backbone.is_eval)

    def test_prompt_expansion_order(self) -> None:
        prompts = expand_prompts(["forest", "river"], ["a satellite photo of {}.", "a remote sensing image of {}."])
        self.assertEqual(
            prompts,
            [
                "a satellite photo of forest.",
                "a remote sensing image of forest.",
                "a satellite photo of river.",
                "a remote sensing image of river.",
            ],
        )

    def test_real_backbone_missing_weights_clear_error(self) -> None:
        backbone = create_backbone("clip_vit_b16", {"backbone": {"name": "clip_vit_b16", "family": "clip", "feature_dim": 512, "weights": None}}, dry_run=False)
        with self.assertRaisesRegex(BackboneUnavailableError, "automatic downloads are disabled"):
            backbone.load_model()

    def test_real_backbone_existing_weights_still_reserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "weights.pt"
            weights.write_text("placeholder\n", encoding="utf-8")
            backbone = create_backbone("clip_vit_b16", {"backbone": {"name": "clip_vit_b16", "family": "clip", "feature_dim": 512, "weights": str(weights)}}, dry_run=False)
            with self.assertRaisesRegex(BackboneUnavailableError, "reserved"):
                backbone.load_model()


if __name__ == "__main__":
    unittest.main()
