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
- Prediction, per-class accuracy, and confusion-matrix CSV exports.
- Guarded table export from raw metrics JSON.
- CPU-only local smoke test.
- Server script templates for later manual remote execution.

Not implemented yet:

- Full CLIP, RemoteCLIP, or GeoRSCLIP feature extraction.
- Full linear probe, Tip-Adapter, Proto-Adapter, or RS-CPC experiments.
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
