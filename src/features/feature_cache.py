from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.timing import utc_now_iso


@dataclass(frozen=True)
class FeatureCache:
    image_features: Any
    image_labels: Any
    image_paths: list[str]
    split_name: str
    class_to_idx: dict[str, int]
    backbone: str
    dataset: str
    feature_dim: int
    normalize_features: bool
    created_at: str
    source_script: str
    text_features: Any | None = None
    text_prompts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> Any:
        return self.image_labels

    @property
    def class_names(self) -> list[str]:
        return class_names_from_mapping(self.class_to_idx)

    def validate(self) -> None:
        image_shape = shape_of_2d(self.image_features)
        label_shape = shape_of_1d(self.image_labels)
        if len(image_shape) != 2:
            raise ValueError("image_features must have shape [num_samples, feature_dim]")
        if len(label_shape) != 1:
            raise ValueError("image_labels must have shape [num_samples]")
        if image_shape[0] != label_shape[0]:
            raise ValueError("image_features and image_labels must have the same first dimension")
        if len(self.image_paths) != image_shape[0]:
            raise ValueError("image_paths length must match image_features")
        if image_shape[1] != self.feature_dim:
            raise ValueError(f"feature_dim={self.feature_dim} does not match image feature width={image_shape[1]}")
        if not self.class_to_idx:
            raise ValueError("class_to_idx must not be empty")
        label_values = to_labels(self.image_labels)
        num_classes = len(self.class_to_idx)
        if any(label < 0 or label >= num_classes for label in label_values):
            raise ValueError("image_labels contain a value outside class_to_idx")
        if self.text_features is not None:
            text_shape = shape_of_2d(self.text_features)
            if len(text_shape) != 2:
                raise ValueError("text_features must have shape [num_classes, feature_dim]")
            if text_shape[0] != num_classes:
                raise ValueError("text feature class count must match class_to_idx")
            if text_shape[1] != self.feature_dim:
                raise ValueError("text_features and image_features must share feature_dim")
        if self.text_prompts is not None and len(self.text_prompts) != num_classes:
            raise ValueError("text_prompts length must match class_to_idx")

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_features": self.image_features,
            "image_labels": self.image_labels,
            "image_paths": self.image_paths,
            "split_name": self.split_name,
            "class_to_idx": self.class_to_idx,
            "text_features": self.text_features,
            "text_prompts": self.text_prompts,
            "backbone": self.backbone,
            "dataset": self.dataset,
            "feature_dim": self.feature_dim,
            "normalize_features": self.normalize_features,
            "created_at": self.created_at,
            "source_script": self.source_script,
            "metadata": self.metadata,
        }


def save_feature_cache(cache: FeatureCache, path: str | Path) -> Path:
    cache.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {destination}")
    with destination.open("wb") as handle:
        pickle.dump(cache.to_dict(), handle)
    return destination


def load_feature_cache(path: str | Path) -> FeatureCache:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Feature cache does not exist: {source}")
    with source.open("rb") as handle:
        data = pickle.load(handle)
    cache = FeatureCache(
        image_features=data["image_features"],
        image_labels=data.get("image_labels", data.get("labels")),
        image_paths=list(data.get("image_paths", [])),
        split_name=str(data.get("split_name", "")),
        class_to_idx=dict(data.get("class_to_idx", {})),
        text_features=data.get("text_features"),
        text_prompts=data.get("text_prompts"),
        backbone=str(data.get("backbone", data.get("metadata", {}).get("backbone", ""))),
        dataset=str(data.get("dataset", data.get("metadata", {}).get("dataset", ""))),
        feature_dim=int(data.get("feature_dim", shape_of_2d(data["image_features"])[1])),
        normalize_features=bool(data.get("normalize_features", False)),
        created_at=str(data.get("created_at", data.get("metadata", {}).get("created_at", ""))),
        source_script=str(data.get("source_script", data.get("metadata", {}).get("source_script", ""))),
        metadata=dict(data.get("metadata", {})),
    )
    cache.validate()
    return cache


def make_fake_feature_cache(
    num_samples: int = 12,
    num_classes: int = 3,
    feature_dim: int = 8,
    seed: int = 1,
    split_name: str = "test",
    dataset: str = "eurosat",
    backbone: str = "fake_backbone",
) -> FeatureCache:
    rng = random.Random(seed)
    class_to_idx = {f"class_{idx}": idx for idx in range(num_classes)}
    labels = [idx % num_classes for idx in range(num_samples)]
    image_features = [normalize([rng.gauss(0.0, 1.0) for _ in range(feature_dim)]) for _ in range(num_samples)]
    text_features = [normalize([rng.gauss(0.0, 1.0) for _ in range(feature_dim)]) for _ in range(num_classes)]
    return FeatureCache(
        image_features=image_features,
        image_labels=labels,
        image_paths=[f"fake://{dataset}/{split_name}/sample_{idx:04d}.jpg" for idx in range(num_samples)],
        split_name=split_name,
        class_to_idx=class_to_idx,
        text_features=text_features,
        text_prompts=[f"a satellite photo of class_{idx}." for idx in range(num_classes)],
        backbone=backbone,
        dataset=dataset,
        feature_dim=feature_dim,
        normalize_features=True,
        created_at=utc_now_iso(),
        source_script="src/features/feature_cache.py",
        metadata={
            "feature_source": "fake_smoke_test",
            "is_real_feature_extraction": False,
            "uses_fake_data": True,
            "uses_fake_features": True,
        },
    )


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


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


def to_labels(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]
