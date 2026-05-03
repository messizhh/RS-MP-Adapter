# Dataset Notes

Supported dataset descriptors:

- `eurosat`
- `aid`
- `nwpu_resisc45`

Dataset roots must come from config files or command-line arguments.

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

```text
AID root/
  AID/                  # common layout, or images/, or class folders directly
    Airport/
      *.jpg|*.jpeg|*.png|*.tif|*.tiff
    Bridge/
    ...
```

```text
NWPU-RESISC45 root/
  NWPU-RESISC45/        # common layout, or images/, or class folders directly
    airplane/
      *.jpg|*.jpeg|*.png|*.tif|*.tiff
    airport/
    ...
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

## Splits

Use `scripts/generate_splits.py` to create deterministic train/val/test and support splits. The split JSON includes dataset root, split policy, ratios, extensions, class count, split sizes, execution environment, run mode, and `is_paper_result`.

Local smoke/tiny split files under `outputs/` are validation artifacts only. Verified real split JSON files under `splits/` may be preserved for reproducibility after review.

Feature extraction should consume verified split files in later server-side phases. Local dry-run feature caches may use fake sample URIs and must not be interpreted as real dataset features.
