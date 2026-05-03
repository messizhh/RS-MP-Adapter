from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pickle
import random
import math



@dataclass(frozen=True)
class FeatureCache:
    image_features: Any
    labels: Any
    class_names: list[str]
    metadata: dict[str, Any]
    text_features: Any | None = None

    def validate(self) -> None:
        image_shape = shape_of_2d(self.image_features)
        label_shape = shape_of_1d(self.labels)
        if len(image_shape) != 2:
            raise ValueError("image_features must have shape [num_samples, feature_dim]")
        if len(label_shape) != 1:
            raise ValueError("labels must have shape [num_samples]")
        if image_shape[0] != label_shape[0]:
            raise ValueError("image_features and labels must have the same first dimension")
        if self.text_features is not None:
            text_shape = shape_of_2d(self.text_features)
            if len(text_shape) != 2:
                raise ValueError("text_features must have shape [num_classes, feature_dim]")
            if text_shape[1] != image_shape[1]:
                raise ValueError("text_features and image_features must share feature_dim")


def save_feature_cache(cache: FeatureCache, path: str | Path) -> Path:
    cache.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {destination}")
    with destination.open("wb") as handle:
        pickle.dump(
            {
            "image_features": cache.image_features,
            "labels": cache.labels,
            "class_names": cache.class_names,
            "metadata": cache.metadata,
            "text_features": cache.text_features,
            },
            handle,
        )
    return destination


def load_feature_cache(path: str | Path) -> FeatureCache:
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    cache = FeatureCache(
        image_features=data["image_features"],
        labels=data["labels"],
        class_names=list(data["class_names"]),
        metadata=dict(data.get("metadata", {})),
        text_features=data.get("text_features"),
    )
    cache.validate()
    return cache


def make_fake_feature_cache(
    num_samples: int = 12,
    num_classes: int = 3,
    feature_dim: int = 8,
    seed: int = 1,
) -> FeatureCache:
    rng = random.Random(seed)
    labels = [idx % num_classes for idx in range(num_samples)]
    image_features = [normalize([rng.gauss(0.0, 1.0) for _ in range(feature_dim)]) for _ in range(num_samples)]
    text_features = [normalize([rng.gauss(0.0, 1.0) for _ in range(feature_dim)]) for _ in range(num_classes)]
    return FeatureCache(
        image_features=image_features,
        labels=labels,
        class_names=[f"class_{idx}" for idx in range(num_classes)],
        text_features=text_features,
        metadata={"feature_source": "fake_smoke_test", "is_real_feature_extraction": False},
    )


def shape_of_1d(value: Any) -> tuple[int, ...]:
    if hasattr(value, "ndim") and hasattr(value, "shape"):
        return tuple(value.shape) if value.ndim == 1 else ()
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return ()


def shape_of_2d(value: Any) -> tuple[int, ...]:
    if hasattr(value, "ndim") and hasattr(value, "shape"):
        return tuple(value.shape) if value.ndim == 2 else ()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        width = len(value[0])
        if all(isinstance(row, (list, tuple)) and len(row) == width for row in value):
            return (len(value), width)
    return ()


def normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]
