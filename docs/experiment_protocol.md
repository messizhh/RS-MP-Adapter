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

`scripts/check_zero_shot_eval_preflight.py` checks whether cached zero-shot evaluation inputs are present for a base split. It reads the feature-cache manifest and train/val/test image caches, verifies image feature/label/path/class-map consistency, and checks text features either from a standalone `text_feature_cache.pt` or from legacy embedded text features in image caches.

The preferred input contract is image features from val/test image caches plus text features from a standalone text feature cache. The standalone text cache is an input artifact, not an evaluation result, and must not be written back into train/val/test image `feature_cache.pt` files. For real cached zero-shot input readiness, dry-run or fake text caches are rejected.

This zero-shot preflight is not zero-shot evaluation. It does not load models, compute logits, compute accuracy, save predictions, train, evaluate, modify image caches, or write `results/raw`. If real standalone text features are missing or malformed, the report sets `zero_shot_input_ready=false` and records recommendations instead of crashing.

Example:

```bash
python3 scripts/check_zero_shot_eval_preflight.py \
  --manifest outputs/manifests/feature_cache_after_seed1_support/feature_cache_manifest.json \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --base-split base_seed1 \
  --text-feature-cache outputs/features/remoteclip_vit_b32/eurosat/base_seed1/eurosat/remoteclip_vit_b32/text/20260512T140232/text_feature_cache.pt \
  --output-dir outputs/preflight/zero_shot_eval \
  --execution-env remote_server \
  --run-mode local_validation
```

`scripts/check_text_feature_cache_preflight.py` checks the standalone text feature cache contract needed before cached zero-shot evaluation can run. It reads the feature-cache manifest and base split, verifies that `class_to_idx` gives a deterministic class order, loads prompt templates from `configs/methods/zero_shot_clip.yaml` or the built-in default, infers the expected feature dimension, and checks whether an independent text cache already exists under the proposed path:

```text
outputs/features/<backbone>/<dataset>/<base_split>/text/text_feature_cache.pt
```

This preflight recommends a separate text feature cache instead of writing `text_features` back into existing train/val/test image caches. It is an input-contract check only: it does not load models, extract text features, compute logits, compute accuracy, save predictions, train, evaluate, modify existing `feature_cache.pt` files, or write `results/raw`.

The proposed text cache schema includes `text_features [num_classes, feature_dim]`, `class_names`, `class_to_idx`, prompts or prompt templates, dataset, backbone, base split, feature dimension, class count, normalization flag, source script, creation time, git commit, execution environment, run mode, and `is_paper_result: false`.

Example:

```bash
python3 scripts/check_text_feature_cache_preflight.py \
  --manifest outputs/manifests/feature_cache_after_seed1_support/feature_cache_manifest.json \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --base-split base_seed1 \
  --output-dir outputs/preflight/text_features \
  --execution-env remote_server \
  --run-mode local_validation
```

`scripts/extract_text_features.py` creates the standalone text feature cache consumed later by cached zero-shot evaluation. It reads the text feature cache preflight report for class order, prompt templates, and expected feature dimension, then writes `text_feature_cache.pt` plus `text_feature_extraction_summary.json` under a timestamped text-cache directory.

This cache is an input artifact, not an evaluation result. It must keep `is_paper_result: false`, must not overwrite train/val/test image `feature_cache.pt` files, and must not write `results/raw`. It does not compute image-text logits, accuracy, predictions, or any training outputs.

Dry-run mode is local-test only: it does not load a model and writes deterministic fake text features with `uses_fake_text_features: true`. Real mode is for server-side use with explicit local backbone weights only; automatic weight downloads remain disabled. For multiple prompt templates, each class gets all templates encoded, then the per-class prompt features are averaged and L2-normalized when the backbone config uses `normalize_features: true`.

Example dry-run:

```bash
python3 scripts/extract_text_features.py \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --base-split base_seed1 \
  --preflight-report outputs/preflight/text_features/eurosat_remoteclip_vit_b32_seed1/20260512T131625/text_feature_cache_preflight_report.json \
  --backbone-config configs/backbones/remoteclip_vit_b32.yaml \
  --method-config configs/methods/zero_shot_clip.yaml \
  --output-dir outputs/features \
  --device cpu \
  --execution-env local_wsl \
  --run-mode local_validation \
  --dry-run
```

Example server real text extraction:

```bash
python3 scripts/extract_text_features.py \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --base-split base_seed1 \
  --preflight-report outputs/preflight/text_features/eurosat_remoteclip_vit_b32_seed1/20260512T131625/text_feature_cache_preflight_report.json \
  --backbone-config configs/backbones/remoteclip_vit_b32.yaml \
  --method-config configs/methods/zero_shot_clip.yaml \
  --weights-path "<REMOTECLIP_WEIGHTS_PATH>" \
  --output-dir outputs/features \
  --device cuda \
  --execution-env remote_server \
  --run-mode server_full
```

`scripts/run_zero_shot.py` is the cached zero-shot evaluation runner. It is the first real evaluation step in the cached-feature workflow: it reads val/test image feature caches plus the standalone text feature cache, computes image-text cosine logits, computes top-1 accuracy, and writes a unique run directory under `results/raw`.

Cached zero-shot evaluation does not load models, extract features, train, tune hyperparameters, or modify image/text caches. Predictions are not saved by default; use `--save-predictions` only when prediction rows are needed for inspection.

Local or debug modes such as `dry_run`, `smoke_test`, `debug`, `tiny_subset`, and `local_validation` are always non-paper results. Paper-facing eligibility requires a server run mode such as `server_full` and explicit `--allow-paper-result`; otherwise `is_paper_result` remains false.

Example cached evaluation command:

```bash
python3 scripts/run_zero_shot.py \
  --config configs/methods/zero_shot_clip.yaml \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --base-split base_seed1 \
  --manifest outputs/manifests/feature_cache_after_seed1_support/feature_cache_manifest.json \
  --text-feature-cache outputs/features/remoteclip_vit_b32/eurosat/base_seed1/eurosat/remoteclip_vit_b32/text/20260512T140232/text_feature_cache.pt \
  --eval-splits val test \
  --preflight-report outputs/preflight/zero_shot_eval/eurosat_remoteclip_vit_b32_seed1/20260512T141833/zero_shot_eval_preflight_report.json \
  --output-dir results/raw \
  --device cpu \
  --execution-env remote_server \
  --run-mode server_full
```

`scripts/check_result_run_preflight.py` is a read-only consistency checker for an existing result run directory. It validates that `metrics.json`, `metadata.json`, a config snapshot, and `log.txt` exist, checks zero-shot-specific fields and metric shapes, and confirms local/debug runs stay excluded from paper-facing tables.

This result preflight does not rerun evaluation, compute logits, recompute accuracy, train, delete files, or modify `metrics.json`/`metadata.json`. It writes only a separate report under `outputs/preflight/result_runs/...`.

Cache-size validation is method-specific. For `zero_shot`, `cache_entries` must equal `num_classes`. For `proto_adapter`, `cache_entries` must equal `num_classes` even when a shot value is present, because it builds one prototype per class. For `tip_adapter`, `cache_entries` must equal `num_classes x shot`; `shot` must be present as a positive integer.

Example:

```bash
python3 scripts/check_result_run_preflight.py \
  --run-dir results/raw/eurosat/remoteclip_vit_b32/zero_shot/seed_1/20260512T143328_5927f59b \
  --expected-method zero_shot \
  --expected-dataset eurosat \
  --expected-backbone remoteclip_vit_b32 \
  --output-dir outputs/preflight/result_runs \
  --execution-env remote_server \
  --run-mode local_validation
```

`scripts/summarize_local_validation_results.py` is an analysis-only read-only summarizer for existing cached local validation runs under `results/raw`. It reads each candidate run's `metrics.json` and `metadata.json`, filters to `run_mode=local_validation`, `is_paper_result=false`, `eligible_for_paper_tables=false`, and the requested dataset/backbone/seed, then writes a timestamped summary under `outputs/analysis/local_validation_summaries/...`.

The local validation summary is not a paper table. It does not run evaluation, compute logits, recompute accuracy, train, modify existing result files, delete result files, or write `results/raw`. If multiple runs exist for the same method/shot/`M`/prototype-init combination, the latest run is selected and the number of candidate runs is recorded.

Example:

```bash
python3 scripts/summarize_local_validation_results.py \
  --results-root results/raw \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --seed 1 \
  --output-dir outputs/analysis/local_validation_summaries \
  --include-methods zero_shot tip_adapter proto_adapter rs_cpc
```

`scripts/check_server_full_protocol_preflight.py` is a read-only protocol gate before formal `server_full` evaluation. It checks the planned dataset/backbone/seed/shot/method matrix, required feature-cache manifests, val/test image caches, support caches, standalone real text caches, adapter input preflights, adapter input plans, and RS-CPC prototype preflight reports.

This protocol preflight does not run experiments, compute logits, recompute accuracy, train, modify existing result files, delete result files, write `results/raw`, or generate paper-facing tables. It separates completed `local_validation` work from future paper-facing server runs. `server_full` does not automatically mean `is_paper_result=true`; formal evaluation commands must explicitly pass `--allow-paper-result`, and the artifact matrix must be complete before any result can be treated as paper-facing.

Example:

```bash
python3 scripts/check_server_full_protocol_preflight.py \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --seeds 1 2 3 \
  --shots 1 2 4 8 16 \
  --methods zero_shot tip_adapter proto_adapter rs_cpc \
  --rs-cpc-prototype-inits mean random_group_mean medoid \
  --rs-cpc-M-values 1 2 4 8 \
  --manifest-template 'outputs/manifests/feature_cache_after_seed{seed}_support/feature_cache_manifest.json' \
  --text-cache-template 'outputs/features/{backbone}/{dataset}/base_seed{seed}/{dataset}/{backbone}/text/*/text_feature_cache.pt' \
  --adapter-plan-template 'outputs/preflight/adapter_input_plans/{dataset}_{backbone}_seed{seed}/*/adapter_input_plan.json' \
  --adapter-preflight-template 'outputs/preflight/adapter_input/{dataset}_{backbone}_seed{seed}/adapter_input_preflight_report.json' \
  --prototype-preflight-template 'outputs/preflight/rs_cpc_prototypes/{dataset}_{backbone}_seed{seed}/*/rs_cpc_prototype_preflight_report.json' \
  --output-dir outputs/preflight/server_full_protocol \
  --execution-env remote_server \
  --run-mode local_validation
```

`scripts/export_seed_expansion_plan.py` reads a server_full protocol preflight report and exports a seed expansion plan for missing seed artifacts. It is a planning artifact only: it does not generate splits, extract image features, extract text features, run adapter or RS-CPC evaluation, train, compute logits, compute accuracy, modify existing result files, or write `results/raw`.

The plan groups work by seed and phase: dataset/split readiness, image feature cache manifest and support caches, standalone text feature cache, adapter input preflight, adapter input plan, RS-CPC prototype preflight, and rerunning the server_full protocol preflight. Suggested commands are templates with TODO placeholders where server paths or exact command choices must be verified before manual execution.

Example:

```bash
python3 scripts/export_seed_expansion_plan.py \
  --server-full-preflight-report outputs/preflight/server_full_protocol/eurosat_remoteclip_vit_b32/20260513T025500/server_full_protocol_preflight_report.json \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --target-seeds 2 3 \
  --shots 1 2 4 8 16 \
  --output-dir outputs/analysis/seed_expansion_plans \
  --execution-env remote_server \
  --run-mode local_validation
```

`scripts/export_seed_execution_command_pack.py` reads a seed expansion plan and exports a per-seed command pack for manual review. The command pack is not an execution script: the generated shell file exits immediately before the command templates, and the commands contain TODO placeholders for server-specific roots, weights, timestamps, and preflight report paths.

This command pack must not be run blindly. It does not generate splits, extract image features, extract text features, evaluate methods, train models, compute logits, compute accuracy, write `results/raw`, modify existing metrics/metadata, or mark anything as a paper result. It is a review artifact to help the user prepare remote-server commands safely.

Example:

```bash
python3 scripts/export_seed_execution_command_pack.py \
  --seed-expansion-plan outputs/analysis/seed_expansion_plans/eurosat_remoteclip_vit_b32_seed2_seed3/20260513T030413/seed_expansion_plan.json \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --seed 2 \
  --shots 1 2 4 8 16 \
  --output-dir outputs/analysis/seed_execution_command_packs \
  --execution-env remote_server \
  --run-mode local_validation
```

## Adapter Input Preflight

`scripts/check_adapter_input_preflight.py` is a read-only preflight for cached-feature adapter inputs. It checks that a feature-cache manifest contains the requested base train/val/test caches and shot support caches, validates cache fields and tensor/list shapes, verifies label/class consistency, and reports expected cache entries for Tip-Adapter, Proto-Adapter, and RS-CPC.

This preflight does not train, tune, evaluate, compute logits, compute accuracy, save predictions, or create paper results. Reports must be written under `outputs/preflight/adapter_input/...`, not `results/raw`.

For RS-CPC, default `M` values are `1, 2, 4, 8`. Each `M` is marked ready only when `M <=` the minimum per-class support count for that shot. Larger `M` values are reported with warnings and `method_input_ready_by_M=false`.

Example:

```bash
python3 scripts/check_adapter_input_preflight.py \
  --manifest outputs/manifests/feature_cache_after_seed1_support/feature_cache_manifest.json \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --base-split base_seed1 \
  --shot-splits shot_1_seed1 shot_2_seed1 shot_4_seed1 shot_8_seed1 shot_16_seed1 \
  --methods tip_adapter proto_adapter rs_cpc \
  --output-dir outputs/preflight/adapter_input \
  --execution-env remote_server \
  --run-mode local_validation
```

`scripts/export_adapter_input_plan.py` derives a machine-readable sweep plan from an adapter input preflight report. It writes JSON and CSV files under `outputs/preflight/adapter_input_plans/...`, with one row per shot for Tip-Adapter and Proto-Adapter and one row per shot-by-`M` for RS-CPC. It is a planning artifact only: it does not load models, train, tune, evaluate, compute logits, compute accuracy, save predictions, or write `results/raw`.

Example:

```bash
python3 scripts/export_adapter_input_plan.py \
  --preflight-report outputs/preflight/adapter_input/eurosat_remoteclip_vit_b32_seed1/adapter_input_preflight_report.json \
  --output-dir outputs/preflight/adapter_input_plans
```

`scripts/run_tip_adapter.py` and `scripts/run_proto_adapter.py` are cached training-free evaluation runners. They consume the adapter input preflight report and adapter input plan, load the shot support image feature cache as the adapter cache input, load val/test image feature caches from the base split, and load the standalone `text_feature_cache.pt` for zero-shot fusion.

Tip-Adapter uses the support image features as the sample-level key-value cache, so `cache_entries = C x shot`. Proto-Adapter builds one mean prototype per class from the same support cache, so `cache_entries = C`. Both runners compute adapter/fused logits, top-1 accuracy, per-split metrics, and a unique run directory under `results/raw/{dataset}/{backbone}/{method}/shot_{shot}/seed_{seed}/{run_id}/`.

These cached training-free adapter evaluations do not train, fine-tune, load CLIP/RemoteCLIP/GeoRSCLIP models, extract features, tune hyperparameters, modify input caches, or generate paper results by default. Predictions are not saved unless `--save-predictions` is passed. Local validation remains `is_paper_result: false` and is excluded from paper-facing tables; even server run modes require explicit `--allow-paper-result`.

Example cached Tip-Adapter local validation command:

```bash
python3 scripts/run_tip_adapter.py \
  --config configs/methods/tip_adapter.yaml \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --shot 16 \
  --base-split base_seed1 \
  --shot-split shot_16_seed1 \
  --manifest outputs/manifests/feature_cache_after_seed1_support/feature_cache_manifest.json \
  --text-feature-cache outputs/features/remoteclip_vit_b32/eurosat/base_seed1/text/20260512T140232/text_feature_cache.pt \
  --adapter-input-plan outputs/preflight/adapter_input_plans/eurosat_remoteclip_vit_b32_seed1/20260512T070522/adapter_input_plan.json \
  --preflight-report outputs/preflight/adapter_input/eurosat_remoteclip_vit_b32_seed1/adapter_input_preflight_report.json \
  --eval-splits val test \
  --output-dir results/raw \
  --device cpu \
  --execution-env local_wsl \
  --run-mode local_validation
```

Proto-Adapter uses the same command shape with `scripts/run_proto_adapter.py` and `configs/methods/proto_adapter.yaml`.

`scripts/check_rs_cpc_prototype_preflight.py` consumes the adapter input plan and source preflight report, then performs RS-CPC prototype construction shape checks only for `method=rs_cpc` rows where `is_ready=true`. It loads support feature caches, constructs temporary prototypes in memory for supported initialization modes, checks prototype shapes and label counts, and writes a JSON report under `outputs/preflight/rs_cpc_prototypes/...`.

This prototype preflight is not an experiment result. It does not load models, use val/test for tuning or evaluation, compute image-to-prototype logits, compute accuracy, save predictions, save prototype tensors, or write `results/raw`. `kmeans` is reserved and may be reported as unsupported for this preflight without failing.

Example:

```bash
python3 scripts/check_rs_cpc_prototype_preflight.py \
  --adapter-input-plan outputs/preflight/adapter_input_plans/eurosat_remoteclip_vit_b32_seed1/20260512T070522/adapter_input_plan.json \
  --preflight-report outputs/preflight/adapter_input/eurosat_remoteclip_vit_b32_seed1/adapter_input_preflight_report.json \
  --prototype-inits mean random_group_mean medoid kmeans \
  --output-dir outputs/preflight/rs_cpc_prototypes \
  --execution-env remote_server \
  --run-mode local_validation
```

`scripts/run_rs_cpc.py` is the cached training-free RS-CPC evaluation runner. It implements the compact multi-prototype cache adapter path: support image features from the shot split are compressed into class-wise prototypes, val/test image features are evaluated against those prototypes, standalone text features provide the zero-shot fusion term, and the runner writes a unique result directory under:

```text
results/raw/{dataset}/{backbone}/rs_cpc/shot_{shot}/M_{M}/{prototype_init}/seed_{seed}/{run_id}/
```

Legal cached RS-CPC combinations follow the adapter input plan and prototype preflight. `M` must be no larger than `shot`. `mean` is valid only for `M=1`. `random_group_mean` and `medoid` are valid when the requested row is ready and `M <= shot`. `kmeans` is reserved and rejected by the cached training-free runner until implemented.

The runner confirms adapter input preflight readiness, confirms the exact `method=rs_cpc` shot-by-`M` row in the adapter input plan, and confirms the exact shot/`M`/`prototype_init` combination in the RS-CPC prototype preflight report. Prototype construction uses only the support cache: val/test caches are not used for prototype construction or hyperparameter selection.

Cached RS-CPC evaluation computes fused logits, top-1 accuracy, per-split metrics, cache entries `C x M`, and efficiency metadata. It does not train, fine-tune, load CLIP/RemoteCLIP/GeoRSCLIP models, extract features, modify input caches, or save predictions unless `--save-predictions` is explicit. Local validation remains `is_paper_result: false` and is excluded from paper-facing tables; server modes also remain non-paper results unless `--allow-paper-result` is explicit.

Example cached RS-CPC local validation command:

```bash
python3 scripts/run_rs_cpc.py \
  --config configs/methods/rs_cpc.yaml \
  --dataset eurosat \
  --backbone remoteclip_vit_b32 \
  --shot 4 \
  --seed 1 \
  --M 2 \
  --prototype-init random_group_mean \
  --manifest outputs/manifests/feature_cache_after_seed1_support/feature_cache_manifest.json \
  --base-split base_seed1 \
  --shot-split shot_4_seed1 \
  --text-feature-cache outputs/features/remoteclip_vit_b32/eurosat/base_seed1/text/20260512T140232/text_feature_cache.pt \
  --adapter-input-plan outputs/preflight/adapter_input_plans/eurosat_remoteclip_vit_b32_seed1/20260512T070522/adapter_input_plan.json \
  --preflight-report outputs/preflight/adapter_input/eurosat_remoteclip_vit_b32_seed1/adapter_input_preflight_report.json \
  --prototype-preflight-report outputs/preflight/rs_cpc_prototypes/eurosat_remoteclip_vit_b32_seed1/rs_cpc_prototype_preflight_report.json \
  --eval-splits val test \
  --output-dir results/raw \
  --device cpu \
  --execution-env local_wsl \
  --run-mode local_validation
```

## Training-Free Method Validation

Phase 1E method runners consume feature caches and can run on fake dry-run caches locally. Local runs must remain `execution_env: local_wsl`, `run_mode: smoke_test`, `device: cpu`, and `is_paper_result: false`.

Implemented local-validation methods are zero-shot, linear probe skeleton, training-free Tip-Adapter, training-free Proto-Adapter, and training-free RS-CPC skeleton. Fine-tuned variants are disabled until later phases.

## Phase 1F Fake Pipeline

Run the local fake end-to-end pipeline with:

```bash
.venv/bin/python scripts/run_fake_pipeline.py \
  --execution-env local_wsl \
  --run-mode smoke_test \
  --device cpu \
  --output-dir outputs/smoke_test/fake_pipeline
```

This pipeline uses synthetic class-folder data and fake feature caches only. It validates dataset inspection, split generation, fake feature extraction, feature-cache validation, zero-shot, linear probe, Tip-Adapter, Proto-Adapter, RS-CPC, and default table exclusion. It does not require real dataset roots or model weights.

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

Server script templates under `scripts/server/` are for future manual remote execution. They contain TODO placeholders for dataset roots, feature roots, checkpoint roots, result roots, and log roots. They must not be run locally and do not indicate that any experiment has completed.

Use AGENTS.md-aligned runtime roots:

- `outputs/features` for feature caches.
- `outputs/checkpoints` for checkpoints and model artifacts.
- `outputs/predictions` for prediction files.
- `results/raw` for raw metrics JSON and run directories.
- `results/tables` for exported CSV/JSON tables.
- `results/figures` for generated figures.
- `results/summaries` for summaries and preflight reports.
- `logs` for logs.
- `splits/{dataset}` for split files.

## Dataset Inspection and Split Generation

Before any real feature extraction or baseline evaluation, inspect the dataset root with `scripts/inspect_dataset.py`. Critical failures include a missing root, no class-folder layout, empty classes, too few samples per class, or a known expected class count mismatch.

Real dataset onboarding is pre-experiment work. The user manually downloads and unzips datasets outside Codex, then provides placeholder-resolved dataset roots. Codex must not download datasets or weights.

Use placeholder variables until real paths are known:

```bash
export RS_DATA_ROOT="<DATA_ROOT>"
export EUROSAT_ROOT="${RS_DATA_ROOT}/EuroSAT"
export AID_ROOT="${RS_DATA_ROOT}/AID"
export NWPU_RESISC45_ROOT="${RS_DATA_ROOT}/NWPU-RESISC45"
```

Expected directory roots:

```text
<EUROSAT_ROOT>/RGB/<class_name>/*                  # or images/<class_name>/, or class folders directly
<AID_ROOT>/AID/<class_name>/*                      # or images/<class_name>/, or class folders directly
<NWPU_RESISC45_ROOT>/NWPU-RESISC45/<class_name>/*  # or images/<class_name>/, or class folders directly
```

Phase 1G adds a read-only dataset layout preflight:

```bash
.venv/bin/python scripts/check_dataset_layout.py \
  --config configs/datasets/eurosat.yaml \
  --dataset eurosat \
  --dataset-root /path/from/user/or/server/config \
  --output-dir outputs/preflight \
  --execution-env local_wsl \
  --run-mode local_validation
```

The preflight writes a timestamped JSON report under `outputs/preflight/{dataset}/` with dataset name, root, class counts, total image count, warnings, per-shot support for 1/2/4/8/16, and `is_ready_for_split_generation`. It is a directory check only; it must not modify datasets, extract features, train, evaluate, download data, or download weights.

Layout preflight commands for all target datasets:

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

- Dataset manually downloaded.
- Dataset manually unzipped.
- Class folders visible.
- Layout preflight passed.
- Preflight JSON reviewed.
- Split generation not yet run unless explicitly requested.
- No feature extraction has been run.
- No training has been run.
- No evaluation has been run.
- No paper result has been produced.

Safe order:

```text
manual dataset placement
-> layout check
-> review report
-> split generation preflight
-> server preflight
-> prepare first real run commands
-> user manually executes server jobs
```

Splits are generated with `scripts/generate_splits.py`. Generation is deterministic by seed and refuses overwrite by default. Use `--overwrite` only for intentional regeneration. Local smoke/tiny split files are marked `is_paper_result: false`; paper-facing splits must come from verified real dataset roots and be preserved.

## Server Dry-Preflight

`scripts/server/check_server_preflight.sh` is a template for later manual server use. It checks Python, PyTorch, CUDA/GPU availability, dataset/feature/checkpoint/result/log root variables, result/log writability, and can optionally call `scripts/check_dataset_layout.py`.

The server preflight is non-experimental. Passing it does not mean experiments are complete or paper-ready; it only means the server appears ready for the next manual setup step.
