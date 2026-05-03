from __future__ import annotations

from pathlib import Path
from typing import Any

from src.features.feature_cache import FeatureCache, class_names_from_mapping, save_feature_cache
from src.models.base_backbone import aggregate_prompt_features, create_backbone, expand_prompts
from src.utils.io import safe_write_json
from src.utils.timing import utc_now_iso


def extract_features_placeholder(*args, **kwargs) -> FeatureCache:
    raise NotImplementedError(
        "Full CLIP/RemoteCLIP/GeoRSCLIP feature extraction is intentionally not implemented in Phase 1D."
    )


def run_dry_run_feature_extraction(
    *,
    dataset: str,
    backbone_name: str,
    backbone_config: dict[str, Any] | None,
    output_dir: str | Path,
    split_path: str | Path | None,
    max_samples: int,
    batch_size: int,
    device: str,
    execution_env: str,
    run_mode: str,
    prompt_templates: list[str] | None = None,
    overwrite: bool = False,
    source_script: str = "scripts/extract_features.py",
) -> dict[str, Any]:
    backbone = create_backbone(backbone_name, backbone_config, dry_run=True, device=device).load_model().eval()
    class_to_idx = {"class_0": 0, "class_1": 1, "class_2": 2}
    class_names = class_names_from_mapping(class_to_idx)
    image_paths = [f"fake://{dataset}/dry_run/sample_{idx:04d}.jpg" for idx in range(max_samples)]
    image_labels = [idx % len(class_to_idx) for idx in range(max_samples)]
    image_features = []
    for start in range(0, max_samples, batch_size):
        batch_paths = image_paths[start : start + batch_size]
        image_features.extend(backbone.encode_images(batch_paths))
    prompts = expand_prompts(class_names, prompt_templates)
    prompt_features = backbone.encode_text(prompts)
    text_features = aggregate_prompt_features(
        prompt_features,
        num_classes=len(class_names),
        templates_per_class=len(prompt_templates or ["a satellite photo of {}."]),
        normalize_features=True,
    )
    cache = FeatureCache(
        image_features=image_features,
        image_labels=image_labels,
        image_paths=image_paths,
        split_name="dry_run",
        class_to_idx=class_to_idx,
        text_features=text_features,
        text_prompts=prompts,
        backbone=backbone.config.name,
        dataset=dataset,
        feature_dim=backbone.get_feature_dim(),
        normalize_features=True,
        created_at=utc_now_iso(),
        source_script=source_script,
        metadata={
            "execution_env": execution_env,
            "run_mode": run_mode,
            "device": device,
            "is_paper_result": False,
            "uses_fake_data": True,
            "uses_fake_features": True,
            "is_real_feature_extraction": False,
            "split_path": str(split_path) if split_path else "",
            "preprocess": backbone.describe_preprocess(),
        },
    )
    cache.validate()
    run_dir = unique_dir(Path(output_dir) / dataset / backbone.config.name / "dry_run")
    cache_path = run_dir / "feature_cache.pt"
    save_feature_cache(cache, cache_path)
    summary = {
        "dataset": dataset,
        "backbone": backbone.config.name,
        "feature_cache_path": str(cache_path),
        "num_images": max_samples,
        "num_classes": len(class_to_idx),
        "feature_dim": backbone.get_feature_dim(),
        "batch_size": batch_size,
        "device": device,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "uses_fake_data": True,
        "uses_fake_features": True,
        "is_real_feature_extraction": False,
        "created_at": utc_now_iso(),
        "source_script": source_script,
    }
    summary_path = safe_write_json(run_dir / "feature_extraction_summary.json", summary, overwrite=overwrite)
    return {"cache": cache, "cache_path": cache_path, "summary_path": summary_path, "run_dir": run_dir}


def save_fake_features_for_smoke(path: str | Path, seed: int = 1) -> Path:
    from src.features.feature_cache import make_fake_feature_cache

    cache = make_fake_feature_cache(seed=seed)
    return save_feature_cache(cache, path)


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique feature extraction directory under {base}")
