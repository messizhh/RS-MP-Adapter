# AGENTS.md v0

## Project: PRICAI Remote Sensing VLM Few-Shot Adaptation

This repository supports a PRICAI 2026 paper project on remote sensing vision-language models, few-shot scene classification, and parameter-efficient adaptation.

Target submission date: 2026-06-13  
Target venue: PRICAI 2026  
Primary tasks: few-shot remote sensing scene classification  
Primary backbones: CLIP, RemoteCLIP, GeoRSCLIP  
Primary datasets: EuroSAT, AID, NWPU-RESISC45  

The intended method is a compact prototype cache adapter for remote sensing VLMs:

- Tentative name: `RS-CPC` or `RS-MP Adapter`
- Full idea: Remote-Sensing Compact Prototype Cache Adapter
- Core goal: reduce cache size and inference cost while preserving accuracy and remote sensing intra-class diversity.

This project must be reproducible, honest, and first-author-driven. Codex is responsible for code implementation, experiment execution, logging, result aggregation, and reproducibility infrastructure. Codex must not invent paper claims or fabricate experimental results.

---

## 1. Roles and General Principles

### 1.1 Role of Codex

Codex should:

- Build and maintain the codebase.
- Implement dataset loaders, split generation, feature extraction, baselines, adapters, evaluators, and result exporters.
- Run experiments only through reproducible scripts and configs.
- Save all logs, configs, checkpoints, metrics, and result files.
- Generate machine-readable result tables from actual experiment outputs.
- Add tests and smoke tests for important modules.

Codex must not:

- Write fake experimental conclusions.
- Manually edit result metrics.
- Insert results into tables unless they were produced by real runs.
- Delete failed runs to hide negative results.
- Overwrite original experiment outputs.
- Present unfinished methods as completed baselines.

### 1.2 Language and Naming

- Use English for code comments, Python identifiers, config keys, filenames, and directory names.
- Chinese is allowed in project notes, planning documents, and high-level research explanations.
- Avoid ambiguous names such as `new_method.py`, `test2.py`, or `final_result.csv`.
- Use explicit names such as `rs_cpc_adapter.py`, `tip_adapter.py`, `generate_splits.py`, and `export_main_table.py`.

### 1.3 Reproducibility Principle

Every experiment must be reproducible from saved files.

Each run must save:

- Config file path and copied config snapshot.
- Seed.
- Dataset name.
- Dataset split file path.
- Shot number.
- Backbone name.
- Method name.
- Command line.
- Git commit hash.
- Python version.
- PyTorch version.
- CUDA version.
- GPU name.
- Start time and end time.
- Log file.
- Checkpoint path, if applicable.
- Result JSON path.
- Result CSV row, if applicable.

### 1.4 Execution Environment and Compute Boundary

Codex runs only on the local host through WSL. Codex must assume that it cannot directly access, schedule, or control the remote GPU server.

The local WSL environment is mainly for:

- Code editing.
- Repository refactoring.
- Config validation.
- Unit tests.
- Smoke tests.
- CPU tests.
- Tiny-subset experiments.
- Feature-shape validation.
- Result-table generation from existing raw results.

Heavy experiments must be designed to run on the remote server, but Codex should only prepare the code, configs, scripts, and commands for those runs. The user will manually execute heavy jobs on the server.

Heavy experiments include:

- Full-dataset feature extraction.
- Full few-shot experiments over all datasets, shots, and seeds.
- Fine-tuning methods such as Tip-Adapter-F, Proto-Adapter-F, and RS-CPC fine-tuned variants.
- Large-backbone experiments using RemoteCLIP or GeoRSCLIP.
- Multi-seed sweeps.
- Ablation sweeps.
- Inference-time benchmarking on large test sets.

Codex must not assume that local WSL results are final paper results unless the run is explicitly marked as a valid full run by the user.

Local WSL runs should be marked as one of:

- `dry_run`
- `smoke_test`
- `debug`
- `tiny_subset`
- `local_validation`

Server runs should be marked as one of:

- `server_full`
- `server_ablation`
- `server_benchmark`

The `execution_env` and `run_mode` fields must be saved in every result JSON and metadata file.

---

## 2. Recommended Repository Structure

Use the following structure unless there is a strong reason to change it.

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   ├── datasets/
│   │   ├── eurosat.yaml
│   │   ├── aid.yaml
│   │   └── nwpu_resisc45.yaml
│   ├── backbones/
│   │   ├── clip_vit_b16.yaml
│   │   ├── remoteclip_vit_b32.yaml
│   │   └── georsclip.yaml
│   ├── methods/
│   │   ├── zero_shot_clip.yaml
│   │   ├── linear_probe.yaml
│   │   ├── tip_adapter.yaml
│   │   ├── tip_adapter_f.yaml
│   │   ├── proto_adapter.yaml
│   │   ├── proto_adapter_f.yaml
│   │   └── rs_cpc.yaml
│   └── experiments/
│       ├── phase1_baselines.yaml
│       └── phase1_rs_cpc.yaml
├── datasets/
│   └── README.md
├── splits/
│   ├── eurosat/
│   ├── aid/
│   └── nwpu_resisc45/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_loader.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── eurosat.py
│   │   ├── aid.py
│   │   ├── nwpu_resisc45.py
│   │   └── split_generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_backbone.py
│   │   ├── remoteclip_backbone.py
│   │   └── georsclip_backbone.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extract_features.py
│   │   └── feature_cache.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── zero_shot.py
│   │   ├── linear_probe.py
│   │   ├── tip_adapter.py
│   │   ├── tip_adapter_f.py
│   │   ├── proto_adapter.py
│   │   └── proto_adapter_f.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── rs_cpc_adapter.py
│   │   ├── fusion_weight.py
│   │   ├── training_free_adapter.py
│   │   └── finetuned_adapter.py
│   ├── prototypes/
│   │   ├── __init__.py
│   │   ├── prototype_builder.py
│   │   ├── cache_compressor.py
│   │   ├── prototype_selector.py
│   │   └── prototype_logits.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   └── per_class_accuracy.py
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── experiment_logger.py
│   │   └── system_info.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── paths.py
│       ├── timing.py
│       └── io.py
├── scripts/
│   ├── generate_splits.py
│   ├── extract_features.py
│   ├── run_zero_shot.py
│   ├── run_linear_probe.py
│   ├── run_tip_adapter.py
│   ├── run_proto_adapter.py
│   ├── run_rs_cpc.py
│   ├── export_tables.py
│   ├── run_smoke_test.py
│   └── server/
│       ├── run_phase1_baselines.sh
│       ├── run_rs_cpc_sweep.sh
│       ├── run_ablation.sh
│       └── export_tables.sh
├── experiments/
│   ├── phase1/
│   └── phase2/
├── outputs/
│   ├── features/
│   ├── checkpoints/
│   └── predictions/
├── logs/
├── results/
│   ├── raw/
│   ├── tables/
│   ├── figures/
│   └── summaries/
├── docs/
│   ├── experiment_protocol.md
│   ├── dataset_notes.md
│   ├── baseline_notes.md
│   └── rs_cpc_notes.md
└── tests/
    ├── test_splits.py
    ├── test_feature_cache.py
    ├── test_metrics.py
    ├── test_prototypes.py
    └── test_smoke.py
```

---

## 3. Dataset and Split Protocol

### 3.1 Main Datasets

The first-stage experiments must support:

1. `EuroSAT`
2. `AID`
3. `NWPU-RESISC45`

Dataset loaders must not assume a single folder format without validation. Each loader should check whether expected image folders, class names, and metadata files exist.

### 3.2 Shot Settings

Few-shot experiments must support:

```text
1, 2, 4, 8, 16
```

The shot number means the number of labeled support images per class.

### 3.3 Seed Settings

Use at least 3 seeds. Prefer 5 seeds when compute allows.

Recommended default seeds:

```text
1, 2, 3, 4, 5
```

For every method, dataset, shot, and backbone combination, the same support set split must be used for fair comparison.

### 3.4 Split Requirements

Each dataset must have fixed:

- Train split.
- Validation split.
- Test split.

Few-shot support sets must be sampled from the train split by class.

Validation split is used for:

- Hyperparameter selection.
- Alpha or beta tuning.
- Early stopping if fine-tuning is used.
- Model selection.

Test split is used only for final evaluation.

### 3.5 Split Storage

All split files must be saved under `splits/{dataset}/`.

Recommended format:

```text
splits/
└── nwpu_resisc45/
    ├── base_split_seed1.json
    ├── base_split_seed2.json
    ├── shot_1_seed1.json
    ├── shot_2_seed1.json
    ├── shot_4_seed1.json
    ├── shot_8_seed1.json
    └── shot_16_seed1.json
```

Each split JSON should include:

```json
{
  "dataset": "nwpu_resisc45",
  "seed": 1,
  "shot": 16,
  "train": [],
  "val": [],
  "test": [],
  "support": [],
  "class_to_idx": {},
  "created_at": "",
  "source_script": "scripts/generate_splits.py"
}
```

Do not resample support sets during training or evaluation unless a new split file is explicitly generated and saved.

---

## 4. Baseline Priority

### 4.1 Phase 1: Required Baselines

The following baselines must be implemented first:

1. Zero-shot CLIP
2. Zero-shot RemoteCLIP
3. Linear probe
4. Tip-Adapter
5. Tip-Adapter-F
6. Proto-Adapter
7. Proto-Adapter-F

These are mandatory for the first complete experimental cycle.

### 4.2 Phase 2: Optional or Interface-Reserved Baselines

The following baselines should be implemented later or have reserved interfaces:

8. TIMO / TIMO-S, if code is usable
9. CLIP-MoA, if code or weights are accessible
10. CRNet, if code or implementation details are accessible
11. AGPL-KEM, if code is released and reproducible
12. Multi-granularity Prompt Learning, if code is available
13. CLIP-Adapter
14. CoOp
15. CoCoOp
16. MaPLe
17. PromptSRC
18. CLIP-LoRA
19. TaskRes
20. LP++
21. APE / APE-T
22. OTAT

If these methods are not reproducible within project resources, they should still be discussed in related work and used to define forbidden claims.

These methods are not first-stage requirements. Do not block Phase 1 implementation on them.

OTAT and other optimal-transport-based methods should be treated as optional baselines or related-work references only. Cross-modal optimal transport alignment must not become the main contribution of this project.

### 4.3 Baseline Implementation Rule

Each baseline should expose a consistent interface:

```python
class MethodBase:
    def fit(self, support_features, support_labels, val_features=None, val_labels=None):
        pass

    def predict_logits(self, image_features):
        pass

    def evaluate(self, image_features, labels):
        pass
```

Training-free methods may implement `fit()` as cache construction or prototype construction.

Fine-tuned methods must save trainable parameters and checkpoints.

---

## 5. Proposed Method: RS-CPC / RS-MP Adapter

### 5.1.1 Method Positioning

The proposed method should not be a generic prompt learning benchmark.

The target contribution is:

```text
A compact multi-prototype cache adapter for remote sensing few-shot scene classification.
```

The method should sit between Tip-Adapter and Proto-Adapter:

- Tip-Adapter uses sample-level key-value cache.
- Proto-Adapter uses one prototype per class.
- RS-CPC / RS-MP uses a small number of compact prototypes per class.

Expected cache size:

```text
Tip-Adapter:   O(CK)
Proto-Adapter: O(C)
RS-CPC:        O(CM), where M << K
```

For NWPU-RESISC45 16-shot:

```text
C = 45
K = 16

Tip-Adapter cache entries:   45 × 16 = 720
Proto-Adapter cache entries: 45
RS-CPC M=2 cache entries:    45 × 2 = 90
RS-CPC M=4 cache entries:    45 × 4 = 180
```

The method should emphasize:

- Cache compactness.
- Inference efficiency.
- Few-shot stability.
- Remote sensing intra-class multi-modal structure.
- Competitive accuracy without relying only on small accuracy gains.

### 5.1.2 Updated Method Scope after 2025–2026 Literature Check

The main method must remain:

**RS-CPC / RS-MP Adapter: Remote-Sensing Compact Prototype Cache Adapter**

The core novelty should be restricted to:

1. Compact prototype-cache construction.
2. Cache compression from `O(CK)` to `O(CM)`, where `M << K`.
3. Preserving remote-sensing intra-class diversity with a small number of prototypes.
4. Improving inference and storage efficiency while maintaining competitive accuracy.
5. Reporting accuracy–cache-size–inference-time tradeoffs.
6. Avoiding transformer-block adapter tuning, prompt learning, OT alignment, and MoA-style adapter gating as the main contribution.

This project is not a prompt-learning paper and not a general adapter-tuning paper. It is a compact prototype-cache adaptation paper for few-shot remote sensing scene classification, emphasizing cache compression, inference efficiency, and preservation of remote-sensing intra-class diversity.


### 5.2 Required Modules

Reserve and implement the following modules:

```text
src/prototypes/prototype_builder.py
src/prototypes/cache_compressor.py
src/prototypes/prototype_selector.py
src/prototypes/prototype_logits.py
src/adapters/fusion_weight.py
src/adapters/training_free_adapter.py
src/adapters/finetuned_adapter.py
src/adapters/rs_cpc_adapter.py
```

### 5.3 Module Responsibilities

#### `prototype_builder`

Responsible for building class-level or multi-prototype representations from support image features.

Required initialization modes:

- `mean`
- `kmeans`
- `random_group_mean`
- `medoid`

#### `cache_compressor`

Responsible for compressing a sample-level cache into a compact prototype-level cache.

Must report:

- Original cache entries.
- Compressed cache entries.
- Compression ratio.
- Number of prototypes per class.
- Runtime of compression.

#### `prototype_selector`

Responsible for selecting or assigning support samples to prototypes.

Should support:

- Class-wise grouping.
- K-means assignment.
- Random grouping with seed control.
- Medoid selection.

#### `prototype_logits`

Responsible for computing logits from image features and prototype cache.

Should support:

- Cosine similarity.
- Temperature scaling.
- Optional margin.
- Optional text calibration.

#### `fusion_weight`

Responsible for combining zero-shot text logits and prototype/cache logits.

Required modes:

- `fixed_alpha`
- `validation_tuned_alpha`
- `adaptive_alpha`

#### `training_free_adapter`

Implements the training-free RS-CPC variant.

#### `finetuned_adapter`

Implements the fine-tuned RS-CPC variant, if enabled.


### 5.4 Optional Non-Main Interfaces

The following optional interfaces may be reserved for baselines, ablations, or future comparison, but they are not part of the main RS-CPC contribution:

```text
src/prototypes/feature_weighting.py
src/prototypes/prototype_refinement.py
src/prototypes/cache_pruning.py
src/prototypes/cache_budget_controller.py
src/baselines/mutual_guidance_baseline.py
src/baselines/prompt_baseline_wrappers/
src/baselines/adapter_baseline_wrappers/
src/adapters/compact_cache/

Rules:

mutual_guidance_baseline is for TIMO-style baseline only, not the main method.
prompt_baseline_wrappers are for CoOp, CoCoOp, MaPLe, PromptSRC, CRNet, AGPL-KEM, and Multi-granularity Prompt baselines only.
adapter_baseline_wrappers are for CLIP-Adapter, CLIP-MoA, OTAT-type baselines only.
Main method code should stay under src/prototypes/ and src/adapters/compact_cache/.
Main method code must not be placed under prompt-learning modules.

### 5.5 Required Config Options

The RS-CPC config must support at least:

```yaml
method: rs_cpc

num_prototypes_per_class: [1, 2, 4, 8]

prototype_init:
  choices: [mean, kmeans, random_group_mean, medoid]

prototype_norm:
  choices: [none, class-wise, channel-wise, both]

fusion:
  choices: [fixed_alpha, validation_tuned_alpha, adaptive_alpha]

use_text_calibration: true
use_margin: false
finetune: false

alpha: 1.0
beta: 1.0
temperature: 1.0
margin: 0.0

save_prototypes: true
save_assignments: true
```

When `finetune: true`, also support:

```yaml
optimizer: adamw
learning_rate: 0.001
weight_decay: 0.0001
epochs: 20
batch_size: 64
early_stopping: true
patience: 5
resume: false
```

---

## 6. Core Experiment Metrics

Every experiment must output the following metrics:

- Top-1 accuracy.
- Mean accuracy over seeds.
- Standard deviation over seeds.
- Cache entries.
- Trainable parameters.
- Training time.
- Inference time.
- Images per second.
- GPU memory, if available.
- Execution environment: `local_wsl`, `remote_server`, or `unknown`.
- Run mode: `dry_run`, `smoke_test`, `debug`, `tiny_subset`, `local_validation`, `server_full`, `server_ablation`, or `server_benchmark`.
- Whether the run is eligible for paper-facing tables.
- Config path.
- Split path.
- Checkpoint path, if applicable.
- Result JSON path.
- Prediction path, if saved.

Recommended raw result JSON schema:

```json
{
  "run_id": "",
  "method": "",
  "backbone": "",
  "dataset": "",
  "shot": 16,
  "seed": 1,
  "execution_env": "local_wsl",
  "run_mode": "smoke_test",
  "is_paper_result": false,
  "host_name": "",
  "device": "cpu",
  "server_job_id": null,
  "top1_acc": 0.0,
  "per_class_acc": {},
  "cache_entries": 0,
  "trainable_params": 0,
  "training_time_sec": 0.0,
  "inference_time_sec": 0.0,
  "images_per_second": 0.0,
  "gpu_memory_mb": null,
  "config_path": "",
  "config_snapshot_path": "",
  "split_path": "",
  "checkpoint_path": null,
  "prediction_path": null,
  "log_path": "",
  "git_commit": "",
  "python_version": "",
  "pytorch_version": "",
  "cuda_version": "",
  "gpu_name": "",
  "command": "",
  "start_time": "",
  "end_time": ""
}
```

---

## 7. Required Result Tables

All tables must be exported from raw JSON or CSV results. Do not manually type metrics into final tables.

### 7.1 Main Accuracy Table

Format:

```text
dataset × shot × method
```

Required columns:

- Dataset.
- Shot.
- Backbone.
- Method.
- Mean top-1 accuracy.
- Standard deviation.
- Number of seeds.
- Result file paths.

### 7.2 Efficiency Table

Format:

```text
method × cache entries × params × train time × inference time
```

Required columns:

- Dataset.
- Shot.
- Backbone.
- Method.
- Cache entries.
- Trainable parameters.
- Training time.
- Inference time.
- Images per second.
- GPU memory.

### 7.3 Cache-Size Tradeoff Table

Compare:

```text
M = 1, 2, 4, 8
```

Required columns:

- Dataset.
- Shot.
- Backbone.
- Number of prototypes per class.
- Cache entries.
- Compression ratio.
- Mean top-1 accuracy.
- Standard deviation.
- Inference time.

### 7.4 Ablation Table

Ablate at least:

- Prototype initialization.
- Text calibration.
- Margin.
- Fusion strategy.
- Prototype normalization.

Required columns:

- Dataset.
- Shot.
- Backbone.
- Method variant.
- Changed component.
- Setting.
- Mean top-1 accuracy.
- Standard deviation.
- Efficiency metrics.

### 7.5 Per-Class Accuracy and Confusion Matrix

Must support:

- Per-class top-1 accuracy.
- Confusion matrix.
- CSV export.
- Figure export.

This is especially important for `NWPU-RESISC45`, where class-level confusion can reveal remote sensing class ambiguity.

### 7.6 Paper Table Filtering

Result table exporters must exclude local/debug runs by default.

Exclude the following `run_mode` values from paper-facing tables unless the user explicitly approves them:

- `dry_run`
- `smoke_test`
- `debug`
- `tiny_subset`
- `local_validation`

Only explicitly approved full runs should be included in paper-facing tables. In normal use, paper-facing tables should include only:

- `server_full`
- `server_ablation`
- `server_benchmark`

The exporter should support options such as:

```bash
--include-run-modes server_full server_ablation server_benchmark
--exclude-run-modes dry_run smoke_test debug tiny_subset local_validation
```

### 7.7 Literature-Driven Experimental Priorities

The updated literature requires the following experimental priorities.

#### Cache Budget Tradeoff

Compare:

- Tip-Adapter
- Proto-Adapter
- RS-CPC with `M = 1, 2, 4, 8`

Report:

- Cache entries.
- Top-1 accuracy mean ± std.
- Inference time.
- Images per second.
- GPU memory, if available.

#### Prompt-Method Separation

All prompt-learning methods must be clearly marked as baselines or related work.

Prompt-learning methods must not be mixed into RS-CPC’s main method.

#### Adapter-Method Separation

CLIP-MoA, OTAT, and other trainable adapter-style methods must not be mixed into RS-CPC’s core method.

They may only appear as optional baselines or related work.

#### Training-Free vs Fine-Tuned Comparison

RS-CPC should support:

- Training-free RS-CPC.
- Optional fine-tuned RS-CPC.

Compare with:

- Tip-Adapter.
- Tip-Adapter-F.
- Proto-Adapter.
- Proto-Adapter-F.
- TIMO / TIMO-S, when available.

#### Efficiency-First Reporting

The paper must not only report accuracy.

Required metrics include:

- Top-1 accuracy mean ± std.
- Cache entries.
- Trainable parameters.
- Training time.
- Inference time.
- Images per second.
- GPU memory, if available.

---

## 8. Phase 1 Codex Task Checklist

Codex should complete the first phase in this order.

### 8.1 Repository Skeleton

- [ ] Create the recommended directory structure.
- [ ] Add `README.md`.
- [ ] Add `requirements.txt` or `environment.yml`.
- [ ] Add placeholder config files.
- [ ] Add minimal package imports.

### 8.2 Config System

- [ ] Implement YAML config loading.
- [ ] Support command-line override.
- [ ] Save config snapshot for each run.
- [ ] Validate required config fields.
- [ ] Ensure all paths come from config.

### 8.3 Local-Server Execution Setup

- [ ] Add `configs/env/local_wsl.yaml`.
- [ ] Add `configs/env/remote_server.yaml`.
- [ ] Add `execution_env` and `run_mode` fields to metadata and result JSON files.
- [ ] Add `--dry-run`, `--max-samples`, `--device`, `--execution-env`, and `--run-mode` arguments to major scripts.
- [ ] Ensure smoke tests can run in local WSL without GPU.
- [ ] Ensure heavy experiment scripts can be generated for remote server execution.
- [ ] Ensure result table export excludes local debug runs by default.

### 8.4 Dataset Loader and Split Generator

- [ ] Implement dataset registry.
- [ ] Implement loaders for EuroSAT, AID, and NWPU-RESISC45.
- [ ] Implement fixed train/val/test split loading.
- [ ] Implement few-shot support split generation.
- [ ] Save split files as JSON or CSV.
- [ ] Add tests for class balance and reproducibility.

### 8.5 CLIP Feature Extraction Cache

- [ ] Implement CLIP/RemoteCLIP/GeoRSCLIP backbone wrappers.
- [ ] Extract image features.
- [ ] Extract text features from class prompts.
- [ ] Save feature cache files.
- [ ] Reuse feature cache when available.
- [ ] Validate feature dimensions and normalization.

### 8.6 Zero-Shot CLIP

- [ ] Implement zero-shot classifier.
- [ ] Support prompt templates.
- [ ] Support CLIP and RemoteCLIP.
- [ ] Evaluate on val and test.
- [ ] Save logits or predictions if configured.

### 8.7 Linear Probe

- [ ] Implement linear probe training on support features.
- [ ] Tune hyperparameters on validation split.
- [ ] Evaluate once on test split.
- [ ] Save checkpoint and result JSON.

### 8.8 Tip-Adapter / Tip-Adapter-F

- [ ] Implement sample-level key-value cache.
- [ ] Implement training-free Tip-Adapter.
- [ ] Implement fine-tuned Tip-Adapter-F.
- [ ] Report cache entries as `C × K`.
- [ ] Save cache tensors and result files.

### 8.9 Proto-Adapter / Proto-Adapter-F

- [ ] Implement one-prototype-per-class cache.
- [ ] Implement training-free Proto-Adapter.
- [ ] Implement fine-tuned Proto-Adapter-F.
- [ ] Report cache entries as `C`.
- [ ] Save prototypes and result files.

### 8.10 RS-CPC Skeleton

- [ ] Implement multi-prototype cache interface.
- [ ] Add config options for `M = 1, 2, 4, 8`.
- [ ] Implement prototype initialization modes.
- [ ] Implement prototype logits.
- [ ] Implement fusion with zero-shot text logits.
- [ ] Save prototype assignments.
- [ ] Report cache entries as `C × M`.

### 8.11 Unified Evaluator and Logger

- [ ] Implement top-1 accuracy.
- [ ] Implement per-class accuracy.
- [ ] Implement confusion matrix.
- [ ] Implement timing.
- [ ] Implement GPU memory logging if available.
- [ ] Save result JSON.
- [ ] Append result CSV without overwriting original files.

### 8.12 Table Generation

- [ ] Implement `scripts/export_tables.py`.
- [ ] Export main accuracy table.
- [ ] Export efficiency table.
- [ ] Export cache-size tradeoff table.
- [ ] Export ablation table.
- [ ] Export per-class accuracy table.
- [ ] Export confusion matrix files.

### 8.13 Tests and Smoke Tests

- [ ] Add unit tests for split generation.
- [ ] Add unit tests for feature cache loading.
- [ ] Add unit tests for metric computation.
- [ ] Add unit tests for prototype construction.
- [ ] Add smoke test using a tiny fake dataset or tiny subset.
- [ ] Ensure smoke test runs quickly on CPU.

---

## 9. Command-Line Standards

All scripts should support:

```bash
--config
--dataset
--backbone
--method
--shot
--seed
--output-dir
--dry-run
--max-samples
--device
--execution-env
--run-mode
```

Training scripts should also support:

```bash
--resume
--checkpoint
```

### 9.1 Generate Splits

```bash
python scripts/generate_splits.py \
  --config configs/datasets/nwpu_resisc45.yaml \
  --dataset nwpu_resisc45 \
  --shots 1 2 4 8 16 \
  --seeds 1 2 3 4 5 \
  --output-dir splits/nwpu_resisc45
```

### 9.2 Extract Features

```bash
python scripts/extract_features.py \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --config configs/backbones/remoteclip_vit_b32.yaml \
  --split splits/nwpu_resisc45/base_split_seed1.json \
  --output-dir outputs/features/nwpu_resisc45/remoteclip_vit_b32
```

### 9.3 Run Zero-Shot CLIP / RemoteCLIP

```bash
python scripts/run_zero_shot.py \
  --config configs/methods/zero_shot_clip.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --split splits/nwpu_resisc45/base_split_seed1.json \
  --seed 1 \
  --output-dir results/raw
```

### 9.4 Run Linear Probe

```bash
python scripts/run_linear_probe.py \
  --config configs/methods/linear_probe.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --shot 16 \
  --split splits/nwpu_resisc45/shot_16_seed1.json \
  --seed 1 \
  --output-dir results/raw
```

### 9.5 Run Tip-Adapter

```bash
python scripts/run_tip_adapter.py \
  --config configs/methods/tip_adapter.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --shot 16 \
  --split splits/nwpu_resisc45/shot_16_seed1.json \
  --seed 1 \
  --output-dir results/raw
```

### 9.6 Run Tip-Adapter-F

```bash
python scripts/run_tip_adapter.py \
  --config configs/methods/tip_adapter_f.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --shot 16 \
  --split splits/nwpu_resisc45/shot_16_seed1.json \
  --seed 1 \
  --finetune \
  --output-dir results/raw
```

### 9.7 Run Proto-Adapter

```bash
python scripts/run_proto_adapter.py \
  --config configs/methods/proto_adapter.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --shot 16 \
  --split splits/nwpu_resisc45/shot_16_seed1.json \
  --seed 1 \
  --output-dir results/raw
```

### 9.8 Run Proto-Adapter-F

```bash
python scripts/run_proto_adapter.py \
  --config configs/methods/proto_adapter_f.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --shot 16 \
  --split splits/nwpu_resisc45/shot_16_seed1.json \
  --seed 1 \
  --finetune \
  --output-dir results/raw
```

### 9.9 Run RS-CPC / RS-MP Adapter

```bash
python scripts/run_rs_cpc.py \
  --config configs/methods/rs_cpc.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --shot 16 \
  --split splits/nwpu_resisc45/shot_16_seed1.json \
  --seed 1 \
  --num-prototypes-per-class 4 \
  --prototype-init kmeans \
  --fusion validation_tuned_alpha \
  --output-dir results/raw
```

### 9.10 Export Result Tables

```bash
python scripts/export_tables.py \
  --input-dir results/raw \
  --output-dir results/tables \
  --tables main efficiency cache_tradeoff ablation per_class
```

### 9.11 Run Smoke Test

```bash
python scripts/run_smoke_test.py \
  --dry-run \
  --run-mode smoke_test \
  --execution-env local_wsl \
  --output-dir outputs/smoke_test
```

### 9.12 Local WSL Commands

Local WSL commands should default to safe, lightweight settings. They are intended for code validation, smoke tests, and tiny-subset debugging, not final paper results.

Example local tiny-subset run:

```bash
python scripts/run_zero_shot.py \
  --config configs/methods/zero_shot_clip.yaml \
  --dataset eurosat \
  --backbone clip_vit_b16 \
  --split splits/eurosat/shot_1_seed1.json \
  --seed 1 \
  --run-mode tiny_subset \
  --execution-env local_wsl \
  --max-samples 128 \
  --device cpu \
  --output-dir results/raw
```

Local commands should support:

- `--dry-run`
- `--max-samples`
- `--num-workers 0`
- `--device cpu`
- `--run-mode`
- `--execution-env`

Local WSL runs should never require large GPU memory.

### 9.13 Server Run Commands

Codex should prepare server-ready commands or shell scripts, but the user will execute them manually on the remote server. Codex must not claim that these experiments have been completed until the user provides result files or logs from the server.

Example server full run:

```bash
python scripts/run_rs_cpc.py \
  --config configs/methods/rs_cpc.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --shot 16 \
  --split splits/nwpu_resisc45/shot_16_seed1.json \
  --seed 1 \
  --num-prototypes-per-class 4 \
  --prototype-init kmeans \
  --fusion validation_tuned_alpha \
  --run-mode server_full \
  --execution-env remote_server \
  --device cuda \
  --output-dir results/raw
```

Server scripts should be saved under:

```text
scripts/server/
```

Recommended server scripts:

```text
scripts/server/run_phase1_baselines.sh
scripts/server/run_rs_cpc_sweep.sh
scripts/server/run_ablation.sh
scripts/server/export_tables.sh
```

---

## 10. Reproducibility and Logging Rules

### 10.1 Required Runtime Metadata

Every experiment must record:

- Git commit hash.
- Python version.
- CUDA version.
- PyTorch version.
- GPU name.
- Host name.
- Execution environment.
- Run mode.
- Device.
- Server job ID, if available.
- Whether the run is eligible for paper-facing tables.
- Command line.
- Config.
- Config snapshot path.
- Seed.
- Dataset.
- Shot.
- Backbone.
- Method.
- Split path.
- Start time.
- End time.
- Result metrics.

### 10.2 Output Directory Rule

Each run should create a unique run directory.

Recommended format:

```text
results/raw/{dataset}/{backbone}/{method}/shot_{shot}/seed_{seed}/{run_id}/
```

Example:

```text
results/raw/nwpu_resisc45/remoteclip_vit_b32/rs_cpc/shot_16/seed_1/20260501_143022_a1b2c3d/
```

Each run directory should contain:

```text
config.yaml
metadata.json
metrics.json
predictions.csv
per_class_accuracy.csv
confusion_matrix.csv
log.txt
checkpoint.pt
prototypes.pt
assignments.json
```

Files that do not apply may be omitted, but `config.yaml`, `metadata.json`, `metrics.json`, and `log.txt` are mandatory.

### 10.3 No Overwrite Rule

Never overwrite original results.

If a result path already exists:

- Create a new `run_id`.
- Or stop with a clear error.
- Do not silently replace previous results.

### 10.4 Result Editing Rule

Do not manually edit result JSON, CSV, or log files.

If a bug is found:

1. Keep the old result.
2. Mark it as invalid in a separate note or metadata field.
3. Fix the code.
4. Rerun the experiment.
5. Save a new result file.

### 10.5 Local-vs-Server Run Metadata

Every run must record where it was executed and whether it is eligible for paper-facing tables.

Required metadata fields:

```json
{
  "execution_env": "local_wsl",
  "run_mode": "smoke_test",
  "is_paper_result": false,
  "host_name": "",
  "device": "cpu_or_cuda",
  "gpu_name": null,
  "server_job_id": null
}
```

Allowed `execution_env` values:

- `local_wsl`
- `remote_server`
- `unknown`

Allowed `run_mode` values:

- `dry_run`
- `smoke_test`
- `debug`
- `tiny_subset`
- `local_validation`
- `server_full`
- `server_ablation`
- `server_benchmark`

Only runs with appropriate full-experiment settings should be considered candidate paper results. In normal use, final tables should prioritize `remote_server` runs with `server_full`, `server_ablation`, or `server_benchmark` mode.

Local WSL results must not be mixed into final paper tables unless explicitly approved and clearly marked.

---

## 11. Code Quality Requirements

### 11.1 Modularity

Code must be modular.

Avoid large scripts that mix:

- Dataset loading.
- Model definition.
- Training.
- Evaluation.
- Logging.
- Table generation.

Use reusable modules under `src/`.

### 11.2 Config-Driven Experiments

Do not hard-code experimental hyperparameters in scripts.

All of the following should come from config or command-line override:

- Dataset root.
- Split path.
- Backbone.
- Method.
- Shot.
- Seed.
- Batch size.
- Learning rate.
- Epochs.
- Alpha/beta.
- Prototype number.
- Prototype initialization.
- Output path.

### 11.3 Randomness Control

Every run must set seeds for:

- Python `random`.
- NumPy.
- PyTorch CPU.
- PyTorch CUDA, if available.

Use deterministic settings when possible.

### 11.4 Train/Eval Separation

Training and evaluation must be logically separated.

- Training code should not evaluate on the test split for hyperparameter selection.
- Validation split should be used for tuning.
- Test split should be used only for final reporting.

### 11.5 Resume Support

Fine-tuned methods should support resume from checkpoint.

Required behavior:

- Load model state.
- Load optimizer state, if available.
- Restore epoch number.
- Continue logging to a new run directory or clearly marked resumed run directory.

### 11.6 Dry Run and Smoke Test

Every major script should support dry run when feasible.

Smoke tests should verify:

- Config loading.
- Dataset registry.
- Split loading.
- Feature cache shape.
- Forward pass.
- Metric computation.
- Result JSON writing.

Smoke tests do not need to produce meaningful accuracy.

### 11.7 Machine-Readable Outputs

All outputs used for analysis must be machine-readable.

Required formats:

- JSON for metadata and metrics.
- CSV for tables and per-class results.
- PT/NPY/NPZ for tensors and feature caches.
- TXT or LOG for human-readable logs.

### 11.8 Local-Server Portability

Code must run both in local WSL and on the remote server.

Do not hard-code machine-specific paths such as:

- `/home/username/...`
- `/mnt/c/...`
- `/data/...`
- `/root/...`

All paths must come from config files or command-line arguments.

Dataset roots, feature-cache roots, output roots, and checkpoint roots must be configurable.

Recommended config fields:

```yaml
paths:
  dataset_root: ""
  split_root: "splits"
  feature_root: "outputs/features"
  checkpoint_root: "outputs/checkpoints"
  result_root: "results/raw"
  log_root: "logs"
```

Local WSL and remote server should use separate machine-specific config override files:

```text
configs/env/local_wsl.yaml
configs/env/remote_server.yaml
```

Do not commit private absolute paths if the repository will be public.

---

## 12. Experiment Protocol

### 12.1 Fair Comparison

For a fair comparison:

- Use the same backbone for all methods within one comparison group.
- Use the same dataset split.
- Use the same support set for each dataset-shot-seed combination.
- Use the same image features when methods are feature-based.
- Use validation split for method-specific hyperparameter selection.
- Report mean and standard deviation over seeds.

### 12.2 Backbone Handling

RemoteCLIP and GeoRSCLIP are backbones or baselines only.

Do not describe them as contributions of this project.

Backbone-specific configs should include:

```yaml
backbone:
  name: remoteclip_vit_b32
  pretrained_path: ""
  image_size: 224
  feature_dim: null
  normalize_features: true
```

### 12.3 Prompt Templates

Zero-shot experiments should support prompt templates, but prompt learning must not become the main contribution.

Prompt templates should be saved in config.

Example:

```yaml
prompts:
  templates:
    - "a satellite photo of a {}."
    - "a remote sensing image of a {}."
    - "an aerial image of a {}."
```

### 12.4 Hyperparameter Selection

Validation split may be used to tune:

- Alpha.
- Beta.
- Temperature.
- Margin.
- Prototype number.
- Fusion strategy.
- Learning rate for fine-tuned variants.

Test split must not be used for hyperparameter selection.

---

## 13. Method Directions to Avoid

This project must avoid the following as main contributions:

1. Generic `CLIP/RemoteCLIP + prompt learning benchmark`.
2. Multi-scale visual prompt or style-statistics prompt as the core idea.
3. Cross-modal optimal transport adapter.
4. Modality gap alignment as the main method.
5. Claiming RemoteCLIP or GeoRSCLIP as a new contribution.
6. Reporting only a small accuracy gain without efficiency analysis.
7. Ignoring cache size, inference time, trainable parameters, and GPU memory.

---

## 13.1 Literature-Aware Constraints / 2025–2026 Update

Recent 2025–2026 literature makes the following directions unsuitable as the main novelty of this project.

### 13.1.1 Additional Forbidden Main Contributions

The project must not claim novelty mainly from:

- Multi-granularity prompt learning.
- Collection-driven prompt learning.
- Resolution-aware prompt learning.
- LLM-generated class collection or commonality prompts.
- Attribute-guided prompt learning.
- Knowledge experts mixture for prompts.
- Expert orthogonality or attribute expert prompt losses.
- Mixture of adapters for multi-task remote sensing classification.
- Visual-branch MoA or gating adapter sharing.
- Text-image mutual guidance in training-free CLIP adaptation.
- Image-guided prompt weighting.
- Text-guided image cache correction.
- General prompt learning for few-shot remote sensing scene classification.

### 13.1.2 Paper-Specific Constraints from Recent Work

Multi-granularity Prompt Learning has already covered few-shot remote sensing scene classification with VLM and multi-granularity prompt learning. Therefore, this project must not use multi-granularity prompt learning as the main contribution.

CRNet has already explored collection-driven and resolution-aware prompt learning, including class collection commonality and resolution-aware visual prompts. Therefore, this project must not claim novelty from collection commonality prompts, resolution-aware prompts, LLM-generated class collection prompts, or resolution-aware visual prompts.

AGPL-KEM has already explored attribute-guided prompt learning with knowledge experts mixture, attribute-specific subspaces, semantic alignment, and expert orthogonality. Therefore, this project must not claim novelty from attribute-guided prompt learning, knowledge expert mixtures, attribute-specific prompt experts, or expert orthogonality prompt losses.

CLIP-MoA has already explored mixture of adapters in the CLIP visual branch for multitask remote sensing classification. Therefore, this project must not claim novelty from visual-branch MoA, gating adapters, multitask adapter sharing, or multi-dataset adapter sharing.

TIMO has already explored training-free text-image mutual guidance for CLIP few-shot adaptation. Therefore, this project must not claim novelty from image-guided text prompt weighting, text-guided image matching correction, or general text-image mutual guidance.

These methods may be discussed as related work or implemented as optional baselines when code and compute resources allow.

## 14. Documentation Requirements

Maintain the following documents:

```text
docs/experiment_protocol.md
docs/dataset_notes.md
docs/baseline_notes.md
docs/rs_cpc_notes.md
```

### 14.1 `experiment_protocol.md`

Should describe:

- Dataset splits.
- Shot settings.
- Seeds.
- Backbone settings.
- Evaluation metrics.
- Result aggregation rules.

### 14.2 `dataset_notes.md`

Should describe:

- Dataset download source.
- Expected folder structure.
- Number of classes.
- Known preprocessing details.
- Any excluded files or corrupted samples.

### 14.3 `baseline_notes.md`

Should describe:

- Implemented baselines.
- Missing baselines.
- External references used for implementation.
- Deviations from original papers, if any.

### 14.4 `rs_cpc_notes.md`

Should describe:

- Prototype construction variants.
- Cache compression behavior.
- Fusion strategy.
- Known limitations.
- Open TODO items.

---

## 15. Minimum Acceptance Criteria for Phase 1

Phase 1 is complete only when the repository can:

- Generate fixed few-shot splits for EuroSAT, AID, and NWPU-RESISC45.
- Extract and cache CLIP or RemoteCLIP features.
- Run zero-shot CLIP or RemoteCLIP.
- Run linear probe.
- Run Tip-Adapter and Tip-Adapter-F.
- Run Proto-Adapter and Proto-Adapter-F.
- Run RS-CPC skeleton with `M = 1, 2, 4, 8`.
- Save raw result JSON files.
- Export main accuracy and efficiency tables.
- Exclude local WSL debug and smoke-test results from paper-facing tables by default.
- Prepare server-run scripts for heavy experiments.
- Run at least one smoke test successfully.
- Preserve all configs, logs, and metadata for each experiment.

---

## 16. Research Constraints

The following constraints are mandatory.

- Do not fabricate experiments.
- Do not fabricate results.
- Do not delete failed results.
- Do not report only the best seed.
- Do not manually modify result files.
- Do not put unrun results into tables.
- Do not describe unimplemented methods as completed.
- Do not claim this work is the first to use CLIP for remote sensing few-shot classification.
- Do not claim RemoteCLIP or GeoRSCLIP as this paper's contribution.
- Do not make prompt learning the main contribution.
- Do not make optimal transport alignment the main contribution.
- Do not hide efficiency metrics when reporting accuracy.
- Do not tune on the test set.
- Do not change splits between methods in the same comparison.
- Do not overwrite original logs, checkpoints, or metrics.
- Do not treat local WSL smoke-test, debug, or tiny-subset runs as final paper results.
- Do not claim server experiments are complete until real server logs and result files are available.
Additional claims to avoid after the 2025–2026 literature update:

- Do not claim to solve remote sensing prompt learning in general.
- Do not claim novelty from multi-granularity prompts.
- Do not claim novelty from resolution-aware or collection-aware prompts.
- Do not claim novelty from LLM-generated class collection or commonality prompts.
- Do not claim novelty from attribute-guided prompts or expert mixtures.
- Do not claim novelty from expert orthogonality or attribute expert prompt losses.
- Do not claim novelty from mixture-of-adapters for multi-task remote sensing classification.
- Do not claim novelty from visual-branch MoA or gating adapter sharing.
- Do not claim novelty from text-image mutual guidance.
- Do not claim novelty from image-guided prompt weighting.
- Do not claim novelty from text-guided image cache correction.
- Do not claim novelty from cross-modal optimal transport alignment.
- Do not claim RemoteCLIP or GeoRSCLIP as the contribution.
- Do not report only the best seed or best accuracy.
- Do not omit efficiency metrics.

When uncertain, preserve the raw evidence and write a note instead of changing or deleting results.

---

## 17. Immediate Next Step for Codex

Start with the following tasks:

1. Create the repository skeleton.
2. Implement the config loader.
3. Add `configs/env/local_wsl.yaml` and `configs/env/remote_server.yaml`.
4. Add `execution_env` and `run_mode` metadata support.
5. Implement dataset registry and split generator.
6. Add smoke tests for split reproducibility.
7. Implement feature cache interface.
8. Implement zero-shot CLIP evaluation.
9. Save all outputs under unique run directories.
10. Prepare server-run script templates, but do not treat them as completed experiments.

Do not start large-scale experiments until the split generator, config snapshot, logger, result JSON writer, and local-vs-server metadata fields are working.
