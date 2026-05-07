from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.features.image_preprocess import inspect_image_metadata, load_rgb_image


class ImagePreprocessTest(unittest.TestCase):
    def test_pil_image_converts_to_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "gray.png"
            Image.new("L", (5, 3), color=128).save(image_path)

            image = load_rgb_image(image_path)

            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (5, 3))

    def test_resize_changes_output_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "rgb.png"
            Image.new("RGB", (8, 4), color=(10, 20, 30)).save(image_path)

            image = load_rgb_image(image_path, image_size=6)

            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (6, 6))

    def test_resize_tuple_changes_output_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "rgb.png"
            Image.new("RGB", (8, 4), color=(10, 20, 30)).save(image_path)

            image = load_rgb_image(image_path, image_size=(7, 5))

            self.assertEqual(image.size, (7, 5))

    def test_metadata_marks_no_model_features_or_paper_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "rgb.png"
            Image.new("RGB", (8, 4), color=(10, 20, 30)).save(image_path)

            metadata = inspect_image_metadata(image_path, image_size=4)

            self.assertEqual(metadata["width"], 4)
            self.assertEqual(metadata["height"], 4)
            self.assertEqual(metadata["mode"], "RGB")
            self.assertTrue(metadata["reads_image_pixels"])
            self.assertFalse(metadata["loads_model"])
            self.assertFalse(metadata["extracts_features"])
            self.assertFalse(metadata["trains_model"])
            self.assertFalse(metadata["evaluates_model"])
            self.assertFalse(metadata["is_paper_result"])


if __name__ == "__main__":
    unittest.main()
