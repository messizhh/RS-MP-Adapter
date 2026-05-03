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

## Inspection

Use `scripts/inspect_dataset.py` with a config and dataset root before generating splits. The report records class counts, sample counts, empty classes, unsupported extensions, configured extension filters, hidden-file policy, and symlink policy.

Inspection must not be treated as an experiment result.

## Splits

Use `scripts/generate_splits.py` to create deterministic train/val/test and support splits. The split JSON includes dataset root, split policy, ratios, extensions, class count, split sizes, execution environment, run mode, and `is_paper_result`.

Local smoke/tiny split files under `outputs/` are validation artifacts only. Verified real split JSON files under `splits/` may be preserved for reproducibility after review.
