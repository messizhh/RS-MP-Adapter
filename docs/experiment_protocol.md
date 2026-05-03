# Experiment Protocol

Local runs are validation runs and are not paper-facing results. Local WSL work is limited to code edits, config validation, unit tests, smoke tests, CPU tests, tiny-subset checks, and feature-shape validation.

## Cached Feature Evaluation

Zero-shot evaluation consumes saved feature caches. A valid cache records image features, image labels, image paths, split name, class mapping, optional text features/prompts, backbone, dataset, feature dimension, normalization flag, creation time, source script, and metadata.

The zero-shot runner computes cosine-similarity logits from cached `image_features` and `text_features`. It does not train a model and does not extract CLIP, RemoteCLIP, or GeoRSCLIP features in local smoke mode.

## Backbone and Feature Extraction Boundary

Backbone wrappers provide a common interface for CLIP, RemoteCLIP, GeoRSCLIP, and fake dry-run backbones. Automatic model weight downloads are disabled. If real weights are requested but unavailable, the code must fail clearly instead of downloading or silently using fake features.

Local WSL feature extraction is limited to `--dry-run` fake feature caches and feature-shape validation. Full feature extraction over real datasets and real backbones is reserved for later server-side execution.

Feature extraction dry-runs write a feature cache and `feature_extraction_summary.json`; both are local validation artifacts with `is_paper_result: false`.

Use `scripts/validate_feature_cache.py` to validate cache schema and dimensions without modifying the original cache.

## Training-Free Method Validation

Phase 1E method runners consume feature caches and can run on fake dry-run caches locally. Local runs must remain `execution_env: local_wsl`, `run_mode: smoke_test`, `device: cpu`, and `is_paper_result: false`.

Implemented local-validation methods are zero-shot, linear probe skeleton, training-free Tip-Adapter, training-free Proto-Adapter, and training-free RS-CPC skeleton. Fine-tuned variants are disabled until later phases.

## Local Smoke Outputs

Smoke outputs may use fake datasets and fake features. They must include:

- `execution_env: local_wsl`
- `run_mode: smoke_test`
- `device: cpu`
- `is_paper_result: false`
- `uses_fake_data: true`
- `uses_fake_features: true`
- `fake_or_dry_run: true`

These outputs cannot be used in paper-facing result tables.

## Table Export Filtering

`scripts/export_tables.py` scans raw `metrics.json` files. By default it excludes `dry_run`, `smoke_test`, `debug`, `tiny_subset`, and `local_validation`, and includes only `server_full`, `server_ablation`, and `server_benchmark`.

If no eligible results exist, it writes empty CSV files with headers and a summary JSON. It must not fabricate rows.

Paper-facing tables must be generated only from explicitly approved server runs.

## Dataset Inspection and Split Generation

Before any real feature extraction or baseline evaluation, inspect the dataset root with `scripts/inspect_dataset.py`. Critical failures include a missing root, no class-folder layout, empty classes, too few samples per class, or a known expected class count mismatch.

Splits are generated with `scripts/generate_splits.py`. Generation is deterministic by seed and refuses overwrite by default. Use `--overwrite` only for intentional regeneration. Local smoke/tiny split files are marked `is_paper_result: false`; paper-facing splits must come from verified real dataset roots and be preserved.
