# RS-MP Adapter

Initial reproducibility infrastructure for a PRICAI 2026 remote sensing VLM few-shot adaptation project.

This repository is scoped around compact prototype-cache adaptation for remote sensing scene classification. Local WSL runs are for code validation, smoke tests, CPU tests, tiny-subset checks, feature-shape validation, and cached local validation only. Local smoke/debug/tiny/local_validation outputs are not paper results.

## Phase 1A Scope

Implemented or available in the current scaffold:

- YAML config loading and overrides.
- Local and remote environment config separation.
- Dataset registry and class-folder split generation.
- Runtime metadata and metrics JSON writing.
- Feature-cache pipeline and shape validation.
- Standalone text feature cache support.
- Cached zero-shot runner.
- Cached training-free Tip-Adapter and Proto-Adapter runners.
- Cached training-free RS-CPC runner.
- Adapter input, text cache, zero-shot eval, and result-run preflight checkers.
- CPU-only local smoke test.
- Server script templates for later manual remote execution.

Not completed as paper-facing work:

- `server_full` experiments across the required datasets, seeds, and backbones.
- Paper-facing result aggregation or final paper tables.
- Linear probe results.
- Fine-tuned Tip-Adapter-F, Proto-Adapter-F, or RS-CPC results.
- Heavy dataset sweeps or paper-facing result generation.

## Current Local Validation Status

`local_validation is not paper result`. The observations below are not eligible for paper-facing tables, do not establish final claims, and must not be described as completed PRICAI paper experiments. They cover only:

| Field | Value |
| --- | --- |
| Dataset | EuroSAT |
| Backbone | RemoteCLIP ViT-B/32 |
| Seed | 1 |
| Feature dimension | 512 |
| Number of classes | 10 |
| Run mode | `local_validation` |
| Paper eligibility | `is_paper_result: false` |

Completed local validation artifacts:

| Artifact | Status |
| --- | --- |
| Image feature cache | `outputs/features/remoteclip_vit_b32/eurosat/base_seed1/` |
| Train features | `16200 x 512` |
| Val features | `5400 x 512` |
| Test features | `5400 x 512` |
| Support caches | `shot_1`, `shot_2`, `shot_4`, `shot_8`, `shot_16` for seed1 |
| Standalone text feature cache | `outputs/features/remoteclip_vit_b32/eurosat/base_seed1/eurosat/remoteclip_vit_b32/text/20260512T140232/text_feature_cache.pt` |
| Preflights | Adapter input, text feature cache, zero-shot eval, and checked result-run preflights passed |

Local validation top-1 observations are rounded to 4 decimals for readability.

| Method | Shot | Val | Test |
| --- | ---: | ---: | ---: |
| Zero-shot | n/a | 0.3170 | 0.3261 |
| Tip-Adapter | 1 | 0.6019 | 0.6061 |
| Tip-Adapter | 2 | 0.7102 | 0.7113 |
| Tip-Adapter | 4 | 0.7361 | 0.7409 |
| Tip-Adapter | 8 | 0.7619 | 0.7670 |
| Tip-Adapter | 16 | 0.8019 | 0.8063 |
| Proto-Adapter | 1 | 0.6085 | 0.6115 |
| Proto-Adapter | 2 | 0.6830 | 0.6763 |
| Proto-Adapter | 4 | 0.7043 | 0.6980 |
| Proto-Adapter | 8 | 0.7217 | 0.7280 |
| Proto-Adapter | 16 | 0.7302 | 0.7322 |

RS-CPC local validation observations use `Val/Test` cells. A dash means that combination has not been reported as completed.

| Init | Shot | M=1 | M=2 | M=4 | M=8 |
| --- | ---: | --- | --- | --- | --- |
| `mean` | 1 | 0.6085/0.6115 | - | - | - |
| `random_group_mean` | 1 | 0.6085/0.6115 | - | - | - |
| `random_group_mean` | 2 | 0.6830/0.6763 | 0.7104/0.7094 | - | - |
| `random_group_mean` | 4 | 0.7043/0.6980 | 0.7287/0.7309 | 0.7304/0.7341 | - |
| `random_group_mean` | 8 | 0.7217/0.7280 | 0.7715/0.7696 | 0.7819/0.7863 | 0.7569/0.7617 |
| `random_group_mean` | 16 | 0.7302/0.7322 | 0.8043/0.8015 | 0.8304/0.8270 | 0.8181/0.8213 |
| `medoid` | 1 | 0.6085/0.6115 | - | - | - |
| `medoid` | 2 | 0.6572/0.6511 | 0.7104/0.7094 | - | - |
| `medoid` | 4 | 0.6280/0.6346 | 0.6396/0.6467 | 0.7304/0.7341 | - |
| `medoid` | 8 | 0.6669/0.6663 | 0.6217/0.6204 | 0.6231/0.6239 | 0.7569/0.7617 |
| `medoid` | 16 | 0.7100/0.7157 | 0.6424/0.6465 | 0.6587/0.6615 | 0.7100/0.7146 |

Current best local_validation observation:

| Method | Init | Shot | M | Val | Test | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RS-CPC | `random_group_mean` | 16 | 4 | 0.8304 | 0.8270 | Local validation only, not a paper result |

Next steps:

- Generate a read-only local_validation summary artifact from existing raw outputs.
- Optionally complete `mean`, `M=1` for shots 2, 4, 8, and 16.
- Define the `server_full` protocol before any paper-facing results are produced.
- Later expand to additional seeds, backbones, and datasets.
- Keep linear probe and fine-tuning work after the cached evaluation protocol is stable.

## Local Smoke Test

```bash
python scripts/run_smoke_test.py \
  --dry-run \
  --run-mode smoke_test \
  --execution-env local_wsl \
  --device cpu \
  --output-dir outputs/smoke_test
```

## Tests

```bash
pytest
```

## Result Policy

Generated local smoke, debug, tiny, or local_validation outputs must include:

- `execution_env: local_wsl`
- `run_mode: smoke_test` or another local-only mode such as `local_validation`
- `is_paper_result: false`
- `device: cpu`

Outputs with `run_mode` equal to `smoke_test`, `dry_run`, `debug`, `tiny_subset`, or `local_validation` cannot enter paper-facing tables. Heavy jobs should be run manually on the remote server using scripts under `scripts/server/`.

Do not manually edit metrics JSON/CSV files. Do not treat local validation observations as final paper claims.
