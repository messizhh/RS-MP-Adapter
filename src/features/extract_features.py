from __future__ import annotations

from pathlib import Path

from src.features.feature_cache import FeatureCache, make_fake_feature_cache, save_feature_cache


def extract_features_placeholder(*args, **kwargs) -> FeatureCache:
    raise NotImplementedError(
        "Full CLIP/RemoteCLIP/GeoRSCLIP feature extraction is intentionally not implemented in Phase 1A."
    )


def save_fake_features_for_smoke(path: str | Path, seed: int = 1) -> Path:
    cache = make_fake_feature_cache(seed=seed)
    return save_feature_cache(cache, path)
