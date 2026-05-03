from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    family: str
    feature_dim: int
    weights: str | None = None
    dry_run: bool = False
    device: str = "cpu"
    normalize_features: bool = True


class BackboneUnavailableError(RuntimeError):
    """Raised when a real backbone is requested without local weights/support."""


class BaseBackbone:
    def __init__(self, config: BackboneConfig) -> None:
        self.config = config
        self.device = config.device
        self.is_loaded = False
        self.is_eval = False

    def load_model(self) -> "BaseBackbone":
        if self.config.dry_run or self.config.family == "fake":
            self.is_loaded = True
            self.is_eval = True
            return self
        if not self.config.weights:
            raise BackboneUnavailableError(
                f"Backbone {self.config.name} requires explicit local weights for real loading; automatic downloads are disabled."
            )
        if not Path(self.config.weights).exists():
            raise BackboneUnavailableError(f"Configured weights do not exist for {self.config.name}: {self.config.weights}")
        raise BackboneUnavailableError(f"Real loading for {self.config.name} is reserved for a later server-side phase.")

    def encode_images(self, images: list[Any]) -> list[list[float]]:
        self._require_loaded()
        if not (self.config.dry_run or self.config.family == "fake"):
            raise BackboneUnavailableError("Real image encoding is not implemented in Phase 1D.")
        return [self._fake_vector(f"image::{item}") for item in images]

    def encode_text(self, prompts: list[str]) -> list[list[float]]:
        self._require_loaded()
        if not (self.config.dry_run or self.config.family == "fake"):
            raise BackboneUnavailableError("Real text encoding is not implemented in Phase 1D.")
        return [self._fake_vector(f"text::{prompt}") for prompt in prompts]

    def get_feature_dim(self) -> int:
        return self.config.feature_dim

    def describe_preprocess(self) -> dict[str, Any]:
        return {
            "mode": "dry_run_fake_preprocess" if self.config.dry_run or self.config.family == "fake" else "configured_backbone_preprocess",
            "device": self.device,
            "normalize_features": self.config.normalize_features,
        }

    def eval(self) -> "BaseBackbone":
        self.is_eval = True
        return self

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(f"Backbone {self.config.name} has not been loaded. Call load_model() first.")

    def _fake_vector(self, key: str) -> list[float]:
        digest = hashlib.sha256(f"{self.config.name}::{key}".encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16)
        rng = random.Random(seed)
        values = [rng.gauss(0.0, 1.0) for _ in range(self.config.feature_dim)]
        return l2_normalize(values) if self.config.normalize_features else values


class FakeBackbone(BaseBackbone):
    def __init__(self, feature_dim: int = 8, device: str = "cpu", name: str = "fake_backbone") -> None:
        super().__init__(BackboneConfig(name=name, family="fake", feature_dim=feature_dim, dry_run=True, device=device))


def build_backbone_config(
    name: str,
    config: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    device: str = "cpu",
) -> BackboneConfig:
    backbone_cfg = (config or {}).get("backbone", config or {})
    family = backbone_cfg.get("family", "fake" if name == "fake_backbone" else name.split("_", 1)[0])
    return BackboneConfig(
        name=backbone_cfg.get("name", name),
        family=family,
        feature_dim=int(backbone_cfg.get("feature_dim", 8 if name == "fake_backbone" else 512)),
        weights=backbone_cfg.get("weights") or backbone_cfg.get("pretrained_path"),
        dry_run=dry_run or name == "fake_backbone",
        device=device,
        normalize_features=bool(backbone_cfg.get("normalize_features", True)),
    )


def create_backbone(name: str, config: dict[str, Any] | None = None, *, dry_run: bool = False, device: str = "cpu") -> BaseBackbone:
    cfg = build_backbone_config(name, config, dry_run=dry_run, device=device)
    if cfg.name == "fake_backbone" or cfg.family == "fake":
        return FakeBackbone(feature_dim=cfg.feature_dim, device=device, name=cfg.name)
    if cfg.family == "clip":
        from src.models.clip_backbone import ClipBackbone

        return ClipBackbone(cfg)
    if cfg.family == "remoteclip":
        from src.models.remoteclip_backbone import RemoteClipBackbone

        return RemoteClipBackbone(cfg)
    if cfg.family == "georsclip":
        from src.models.georsclip_backbone import GeoRsClipBackbone

        return GeoRsClipBackbone(cfg)
    return BaseBackbone(cfg)


def expand_prompts(class_names: list[str], templates: list[str] | None = None) -> list[str]:
    prompt_templates = templates or ["a satellite photo of {}."]
    prompts = []
    for class_name in class_names:
        for template in prompt_templates:
            prompts.append(template.format(class_name))
    return prompts


def aggregate_prompt_features(
    prompt_features: list[list[float]],
    num_classes: int,
    templates_per_class: int,
    normalize_features: bool = True,
) -> list[list[float]]:
    if templates_per_class <= 0:
        raise ValueError("templates_per_class must be positive")
    if len(prompt_features) != num_classes * templates_per_class:
        raise ValueError("Prompt feature count does not match num_classes * templates_per_class")
    aggregated = []
    for class_idx in range(num_classes):
        chunk = prompt_features[class_idx * templates_per_class : (class_idx + 1) * templates_per_class]
        mean = [sum(values) / templates_per_class for values in zip(*chunk)]
        aggregated.append(l2_normalize(mean) if normalize_features else mean)
    return aggregated


def l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]
