# RS-MP Adapter

Initial reproducibility infrastructure for a PRICAI 2026 remote sensing VLM few-shot adaptation project.

This repository is scoped around compact prototype-cache adaptation for remote sensing scene classification. Local WSL runs are for code validation, smoke tests, CPU tests, tiny-subset checks, and feature-shape validation only. Local smoke/debug/tiny outputs are not paper results.

## Current Scope

Implemented in this scaffold:

- YAML config loading and overrides.
- Local and remote environment config separation.
- Dataset registry and class-folder split generation.
- Runtime metadata and metrics JSON writing.
- Feature-cache interfaces with schema and shape validation.
- Zero-shot evaluation over cached features.
- Backbone wrapper interfaces with dry-run fake feature support.
- Local dry-run feature extraction and feature-cache validation.
- Feature-cache-based training-free baseline skeletons.
- Prediction, per-class accuracy, and confusion-matrix CSV exports.
- Guarded table export from raw metrics JSON.
- CPU-only local smoke test.
- Server script templates for later manual remote execution.

Not implemented yet:

- Full CLIP, RemoteCLIP, or GeoRSCLIP feature extraction.
- Full linear probe, Tip-Adapter, Proto-Adapter, or RS-CPC experiments.
- Fine-tuned variants such as Tip-Adapter-F, Proto-Adapter-F, and fine-tuned RS-CPC.
- Heavy dataset sweeps or paper-facing result generation.

## Local Smoke Test

Use `python3` on WSL systems where `python` is not on `PATH`:

```bash
python3 scripts/run_smoke_test.py \
  --dry-run \
  --run-mode smoke_test \
  --execution-env local_wsl \
  --device cpu \
  --output-dir outputs/smoke_test
```

## Local Fake Pipeline

Phase 1F includes a local fake end-to-end pipeline for CPU validation only:

```bash
.venv/bin/python scripts/run_fake_pipeline.py \
  --execution-env local_wsl \
  --run-mode smoke_test \
  --device cpu \
  --output-dir outputs/smoke_test/fake_pipeline
```

The pipeline creates a synthetic class-folder dataset, inspects it, generates fake splits, extracts fake features, validates the feature cache, runs zero-shot, linear probe, Tip-Adapter, Proto-Adapter, and RS-CPC training-free skeletons, then exports tables and verifies that default table filtering excludes all smoke/fake/local rows.

All local fake pipeline outputs are validation artifacts with `is_paper_result: false`, `uses_fake_data: true`, and `uses_fake_features: true`. They are not experimental results.

## Tests

```bash
python3 -m unittest discover -s tests
python3 -m pytest
```

`pytest` is declared in `requirements.txt` and `environment.yml`. If `python3 -m pytest` fails because pytest is not installed, install project dependencies in a virtual environment or user environment, for example:

```bash
python3 -m pip install --user -r requirements.txt
```

## Feature Cache Schema

Feature cache files are machine-readable tensor/list caches saved through `src/features/feature_cache.py`. They contain:

- `image_features`, `image_labels`, `image_paths`
- `split_name`, `class_to_idx`
- optional `text_features`, optional `text_prompts`
- `backbone`, `dataset`, `feature_dim`
- `normalize_features`, `created_at`, `source_script`
- metadata flags such as `uses_fake_data` and `uses_fake_features`

Local smoke tests create fake feature caches only. Real cached-feature evaluation must provide `--feature-cache`; missing caches fail clearly.

## Backbone and Feature Extraction

Backbone wrappers expose a common interface:

- `load_model()`
- `encode_images(...)`
- `encode_text(...)`
- `get_feature_dim()`
- `describe_preprocess()`

Automatic model weight downloads are disabled. Real CLIP, RemoteCLIP, and GeoRSCLIP loading requires explicit local weights and will be enabled in a later server-side phase. Local WSL validation should use dry-run fake features only:

```bash
.venv/bin/python scripts/extract_features.py \
  --dataset eurosat \
  --backbone fake_backbone \
  --dry-run \
  --max-samples 12 \
  --batch-size 4 \
  --device cpu \
  --execution-env local_wsl \
  --run-mode smoke_test \
  --output-dir outputs/smoke_test/features
```

Validate a cache without modifying it:

```bash
.venv/bin/python scripts/validate_feature_cache.py \
  --feature-cache outputs/smoke_test/features/.../feature_cache.pt \
  --output-dir outputs/smoke_test/feature_cache_validation \
  --execution-env local_wsl \
  --run-mode smoke_test
```

Local dry-run feature caches are not paper results.

## Cached Zero-Shot Workflow

```bash
python3 scripts/run_zero_shot.py \
  --config configs/methods/zero_shot_clip.yaml \
  --dataset eurosat \
  --backbone clip_vit_b16 \
  --feature-cache outputs/features/example.pt \
  --split splits/eurosat/shot_1_seed1.json \
  --seed 1 \
  --execution-env local_wsl \
  --run-mode tiny_subset \
  --device cpu \
  --output-dir results/raw
```

This command evaluates logits from cached `image_features` and `text_features`; it does not extract CLIP features or train a model.

## Result Policy

Generated local smoke outputs must include:

- `execution_env: local_wsl`
- `run_mode: smoke_test` or another local-only mode
- `is_paper_result: false`
- `device: cpu`

Outputs with `run_mode` equal to `smoke_test`, `dry_run`, `debug`, `tiny_subset`, or `local_validation` cannot enter paper-facing tables. Fake-feature smoke metrics include explicit fake-data flags and must not be interpreted as real zero-shot accuracy.

`scripts/export_tables.py` excludes local/debug/smoke/tiny/local-validation runs by default and includes only `server_full`, `server_ablation`, and `server_benchmark`. If no eligible results exist, it writes empty CSV files with headers plus a summary JSON; it never fabricates rows.

Server scripts under `scripts/server/` are templates only. Before future server use, the user must manually fill TODO placeholders for dataset roots, feature roots, weight roots, and output roots on the remote server. Do not execute these templates locally.

## Training-Free Cached-Feature Methods

Phase 1E provides CPU-safe method interfaces and fake-cache validation for:

- zero-shot cached evaluation
- linear probe skeleton using a nearest-centroid fallback
- training-free Tip-Adapter
- training-free Proto-Adapter
- training-free RS-CPC compact prototype-cache skeleton

These implementations operate on feature caches. Local runs should use `--dry-run` or explicitly provided fake/tiny caches only. Fine-tuned variants are intentionally disabled and raise clear errors. RS-CPC remains compact prototype-cache adaptation; it is not prompt learning and does not use optimal transport alignment.

## Dataset Inspection

Expected dataset layout is class-folder based. The loader searches configured candidate roots and then expects one directory per class:

```text
EuroSAT root/
  RGB/ or images/ or ./
    AnnualCrop/
    Forest/

AID root/
  AID/ or images/ or ./
    Airport/
    Bridge/

NWPU-RESISC45 root/
  NWPU-RESISC45/ or images/ or ./
    airplane/
    airport/
```

Inspect a dataset root before generating official splits:

```bash
.venv/bin/python scripts/inspect_dataset.py \
  --config configs/datasets/eurosat.yaml \
  --dataset eurosat \
  --dataset-root /path/from/user/or/server/config \
  --output-dir outputs/dataset_inspection \
  --execution-env local_wsl \
  --run-mode local_validation \
  --write-report
```

Dataset roots must come from config or CLI. Do not hard-code private paths.

## Real Dataset Onboarding

This is a pre-experiment onboarding workflow. It does not download data, extract features, train, evaluate, or produce paper results.

The user should manually download and unzip datasets outside Codex, then provide placeholder-resolved roots:

```bash
export RS_DATA_ROOT="<DATA_ROOT>"
export EUROSAT_ROOT="${RS_DATA_ROOT}/EuroSAT"
export AID_ROOT="${RS_DATA_ROOT}/AID"
export NWPU_RESISC45_ROOT="${RS_DATA_ROOT}/NWPU-RESISC45"
```

Expected roots after manual placement:

```text
<EUROSAT_ROOT>/RGB/<class_name>/*          # or images/<class_name>/, or class folders directly
<AID_ROOT>/AID/<class_name>/*              # or images/<class_name>/, or class folders directly
<NWPU_RESISC45_ROOT>/NWPU-RESISC45/<class_name>/*
                                           # or images/<class_name>/, or class folders directly
```

Run read-only layout checks before any split generation:

```bash
.venv/bin/python scripts/check_dataset_layout.py \
  --config configs/datasets/eurosat.yaml \
  --dataset eurosat \
  --dataset-root "<EUROSAT_ROOT>" \
  --output-dir outputs/preflight \
  --execution-env local_wsl \
  --run-mode local_validation
```

```bash
.venv/bin/python scripts/check_dataset_layout.py \
  --config configs/datasets/aid.yaml \
  --dataset aid \
  --dataset-root "<AID_ROOT>" \
  --output-dir outputs/preflight \
  --execution-env local_wsl \
  --run-mode local_validation
```

```bash
.venv/bin/python scripts/check_dataset_layout.py \
  --config configs/datasets/nwpu_resisc45.yaml \
  --dataset nwpu_resisc45 \
  --dataset-root "<NWPU_RESISC45_ROOT>" \
  --output-dir outputs/preflight \
  --execution-env local_wsl \
  --run-mode local_validation
```

First-experiment preflight checklist:

- Dataset was manually downloaded and unzipped by the user.
- Class folders are visible.
- Layout preflight passed.
- Preflight JSON was reviewed.
- Split generation has not been run unless explicitly requested.
- No feature extraction, training, or evaluation has been run.

Safe order: manual dataset placement -> layout check -> review report -> split generation preflight -> server preflight -> prepare first real run commands -> user manually executes server jobs.

Before generating real split files, run the read-only layout preflight:

```bash
.venv/bin/python scripts/check_dataset_layout.py \
  --config configs/datasets/eurosat.yaml \
  --dataset eurosat \
  --dataset-root /path/from/user/or/server/config \
  --output-dir outputs/preflight \
  --execution-env local_wsl \
  --run-mode local_validation
```

This writes a timestamped JSON report under `outputs/preflight/{dataset}/` and checks class folders, image counts, empty classes, non-image files, duplicate or invalid class names, and whether 1/2/4/8/16-shot splits are supported. It is read-only and does not train, evaluate, extract features, download data, or download weights.

For later remote onboarding, `scripts/server/check_server_preflight.sh` is a template for manual server use. It checks Python/PyTorch/CUDA/GPU availability, TODO dataset/feature/weight/output roots, output writability, and can optionally call `scripts/check_dataset_layout.py`. It is not an experiment script.

## Split Generation

Generate fixed splits only after dataset inspection passes:

```bash
.venv/bin/python scripts/generate_splits.py \
  --config configs/datasets/eurosat.yaml \
  --dataset eurosat \
  --dataset-root /path/from/user/or/server/config \
  --shots 1 2 4 8 16 \
  --seeds 1 2 3 4 5 \
  --output-dir splits/eurosat
```

Split generation refuses overwrite by default. Use `--overwrite` only when intentionally replacing generated split JSON files. Local smoke/tiny split files are not paper results; verified real paper-facing splits should be preserved.

Do not manually edit metrics JSON/CSV files. Heavy jobs should be run manually on the remote server using scripts under `scripts/server/`.

Fake smoke splits are generated under `outputs/` and are ignored by Git. Real official split JSON files under `splits/` may be committed later after review.
