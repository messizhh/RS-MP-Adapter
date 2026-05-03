# Experiment Protocol

Local runs are validation runs and are not paper-facing results. Local WSL work is limited to code edits, config validation, unit tests, smoke tests, CPU tests, tiny-subset checks, and feature-shape validation.

## Cached Feature Evaluation

Zero-shot evaluation consumes saved feature caches. A valid cache records image features, image labels, image paths, split name, class mapping, optional text features/prompts, backbone, dataset, feature dimension, normalization flag, creation time, source script, and metadata.

The zero-shot runner computes cosine-similarity logits from cached `image_features` and `text_features`. It does not train a model and does not extract CLIP, RemoteCLIP, or GeoRSCLIP features in local smoke mode.

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
