# Dataset Notes

Supported dataset descriptors:

- `eurosat`
- `aid`
- `nwpu_resisc45`

Dataset roots must come from config files or command-line arguments.

## Real Dataset Onboarding Status

This is pre-experiment onboarding. It prepares dataset directories for later checks; it is not feature extraction, training, evaluation, or a paper-result step.

Do not download datasets from inside this repository workflow. The user should manually obtain the datasets from their official or institution-approved sources, then place and unzip them under a local or server path chosen by the user. Use placeholder paths in docs and scripts until real paths are known.

Example placeholder root variables:

```bash
export RS_DATA_ROOT="<DATA_ROOT>"
export EUROSAT_ROOT="${RS_DATA_ROOT}/EuroSAT"
export AID_ROOT="${RS_DATA_ROOT}/AID"
export NWPU_RESISC45_ROOT="${RS_DATA_ROOT}/NWPU-RESISC45"
```

Manual placement outline:

```text
1. Manually download EuroSAT, AID, and NWPU-RESISC45 outside Codex.
2. Place the archives under <DATA_ARCHIVE_ROOT>/.
3. Unzip each archive under <DATA_ROOT>/.
4. Confirm class folders are visible before running any split generation.
5. Run read-only layout preflight and review the JSON report.
```

No split generation should be run until the layout preflight passes and the report has been reviewed.

## Folder Layouts

The Phase 1C loaders support class-folder datasets. The configured candidates are:

- EuroSAT: `images/`, `RGB/`, or the dataset root itself.
- AID: `AID/`, `images/`, or the dataset root itself.
- NWPU-RESISC45: `NWPU-RESISC45/`, `images/`, or the dataset root itself.

Each class directory should contain image files with configured extensions. Hidden files and hidden directories are ignored by default, symlinks are not followed by default, and unsupported extensions are reported by inspection.

Expected layouts:

```text
EuroSAT root/
  RGB/                  # common layout, or images/, or class folders directly
    AnnualCrop/
      *.jpg|*.jpeg|*.png|*.tif|*.tiff
    Forest/
    ...
```

Alternative accepted EuroSAT layouts:

```text
<EUROSAT_ROOT>/images/<class_name>/*
<EUROSAT_ROOT>/<class_name>/*
```

```text
AID root/
  AID/                  # common layout, or images/, or class folders directly
    Airport/
      *.jpg|*.jpeg|*.png|*.tif|*.tiff
    Bridge/
    ...
```

Alternative accepted AID layouts:

```text
<AID_ROOT>/images/<class_name>/*
<AID_ROOT>/<class_name>/*
```

```text
NWPU-RESISC45 root/
  NWPU-RESISC45/        # common layout, or images/, or class folders directly
    airplane/
      *.jpg|*.jpeg|*.png|*.tif|*.tiff
    airport/
    ...
```

Alternative accepted NWPU-RESISC45 layouts:

```text
<NWPU_RESISC45_ROOT>/images/<class_name>/*
<NWPU_RESISC45_ROOT>/<class_name>/*
```

Configured expected class counts:

- EuroSAT: 10 classes.
- AID: 30 classes.
- NWPU-RESISC45: 45 classes.

Class folder names should be stable, non-empty names using letters, numbers, spaces, underscores, hyphens, or periods. Duplicate class names after case-insensitive normalization should be fixed before split generation.

## Inspection

Use `scripts/inspect_dataset.py` with a config and dataset root before generating splits. The report records class counts, sample counts, empty classes, unsupported extensions, configured extension filters, hidden-file policy, and symlink policy.

Inspection must not be treated as an experiment result.

## Layout Preflight

Before generating real splits, run the read-only layout preflight:

```bash
.venv/bin/python scripts/check_dataset_layout.py \
  --config configs/datasets/eurosat.yaml \
  --dataset eurosat \
  --dataset-root /path/from/user/or/server/config \
  --output-dir outputs/preflight \
  --execution-env local_wsl \
  --run-mode local_validation
```

The preflight checks root existence, candidate class roots, class folders, expected class count, image counts per class, empty classes, unsupported/non-image files, duplicate or invalid class names, and support for 1/2/4/8/16-shot split generation from the configured train ratio. It writes a timestamped JSON report under `outputs/preflight/{dataset}/`.

The preflight does not modify the dataset, extract features, train, evaluate, download data, or download weights.

Run one command per dataset after manual placement:

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

Review each JSON report before continuing. Required review points:

- `root_exists: true`
- `root_is_dir: true`
- `num_classes` matches the configured expected class count.
- `empty_classes` is empty.
- `invalid_class_names` is empty.
- `duplicate_class_names` is empty.
- `supports_shots` is true for `1`, `2`, `4`, `8`, and `16`.
- `is_ready_for_split_generation: true`

## First-Experiment Preflight Checklist

Complete this checklist before requesting split generation or first real run command preparation:

- Dataset manually downloaded by the user.
- Dataset archive manually unzipped under a placeholder-resolved data root.
- Class folders are visible for EuroSAT, AID, and NWPU-RESISC45.
- `scripts/check_dataset_layout.py` passed for the target dataset.
- The generated preflight JSON has been reviewed.
- Split generation has not been run unless explicitly requested.
- No feature extraction has been run.
- No training has been run.
- No evaluation has been run.
- No paper result has been produced.

Safe order:

```text
manual dataset placement
-> layout check
-> review preflight report
-> split generation preflight
-> server preflight
-> prepare first real run commands
-> user manually executes server jobs
```

## Splits

Use `scripts/generate_splits.py` to create deterministic train/val/test and support splits. The split JSON includes dataset root, split policy, ratios, extensions, class count, split sizes, execution environment, run mode, and `is_paper_result`.

Local smoke/tiny split files under `outputs/` are validation artifacts only. Verified real split JSON files under `splits/` may be preserved for reproducibility after review.

Feature extraction should consume verified split files in later server-side phases. Local dry-run feature caches may use fake sample URIs and must not be interpreted as real dataset features.
