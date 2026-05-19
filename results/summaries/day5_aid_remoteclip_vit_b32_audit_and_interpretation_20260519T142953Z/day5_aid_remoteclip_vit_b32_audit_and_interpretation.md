# Day5 AID + RemoteCLIP Tables and Audit

Generated: 2026-05-19T14:29:53Z

Source table package:

`results/tables/day4_aid_remoteclip_vit_b32_20260519T135258Z_1`

Status: candidate verified analysis tables only. These are not final paper tables, and no row is marked as a final paper result.

## Scope

This Day5 audit only reads and interprets the lightweight table package files already present in the repository:

- `day4_completion_summary.json`
- `day4_completion_summary.md`
- `day4_table_package_sha256_excluding_manifest.txt`
- `day2_table_audit_summary.json`
- `day2_table_audit_summary.md`
- `inclusion_registry.csv`
- `inclusion_registry.json`
- `main_accuracy_summary.csv`
- `main_accuracy_seed_rows.csv`
- `efficiency_summary.csv`
- `efficiency_seed_rows.csv`
- `cache_tradeoff_summary.csv`
- `cache_tradeoff_seed_rows.csv`
- `rs_cpc_m_prototype_init_ablation_summary.csv`
- `rs_cpc_m_prototype_init_ablation_seed_rows.csv`

No new experiments were run. `results/raw` was not read or modified. `AGENTS.md` was not modified, staged, or committed.

The local lightweight git package does not include:

- `confusion_matrix_seed_rows.csv`
- `per_class_accuracy_seed_rows.csv`

The Day4 audit metadata says those files were available in the original full table export, and the SHA256 manifest still contains entries for them. They are not present in this local checkout, so Day5 does not audit their contents. This is not treated as a Day5 blocker because the current repository intentionally carries a lightweight package.

## Integrity Audit

The Day4 completion summary confirms:

| Check | Value |
| --- | ---: |
| `expected_num_runs` | 132 |
| `found_num_runs_with_metadata_and_metrics` | 132 |
| `missing_num_runs` | 0 |
| `num_whitelist_rows` | 132 |
| `num_included` / included rows | 132 |
| `num_excluded` / excluded rows | 0 |
| included execution envs | `remote_server` only |
| included run modes | `server_full` only |
| paper-facing status | `not_marked_as_final_paper_tables` |

The table audit summary independently reports:

- `num_preflight_rows`: 132
- `num_included`: 132
- `num_excluded`: 0
- `included_execution_envs`: `remote_server`
- `included_run_modes`: `server_full`
- `does_not_modify_results_raw`: true
- `does_not_run_experiments`: true
- `does_not_scan_raw_root`: true

Raw paper-result flags are all false:

- `metadata_is_paper_result_true`: 0
- `metrics_is_paper_result_true`: 0
- `metadata_eligible_for_paper_tables_true`: 0
- `metrics_eligible_for_paper_tables_true`: 0

The inclusion registry has 132 rows:

| Field | Distribution |
| --- | --- |
| dataset | `aid`: 132 |
| backbone | `remoteclip_vit_b32`: 132 |
| execution env | `remote_server`: 132 |
| run mode | `server_full`: 132 |
| raw metrics paper result | `False`: 132 |
| raw metadata paper result | `False`: 132 |
| raw metrics eligible for paper tables | `False`: 132 |
| raw metadata eligible for paper tables | `False`: 132 |
| policy status | `pending_final_policy`: 132 |

The local SHA256 check against `day4_table_package_sha256_excluding_manifest.txt` found:

- 14 manifest entries.
- 12 local manifest-listed files present and hash-matched.
- 2 manifest-listed files missing locally: `confusion_matrix_seed_rows.csv` and `per_class_accuracy_seed_rows.csv`.
- 0 hash mismatches among locally present files.

The missing large files are an artifact-package scope issue, not evidence of result inconsistency in the lightweight files.

## Coverage and Anomaly Check

Seed-row coverage is internally consistent:

| File | Rows | Seed coverage | Bad seed groups |
| --- | ---: | --- | ---: |
| `main_accuracy_seed_rows.csv` | 132 | seeds 1, 2, 3, each 44 rows | 0 |
| `efficiency_seed_rows.csv` | 132 | seeds 1, 2, 3, each 44 rows | 0 |
| `cache_tradeoff_seed_rows.csv` | 129 | seeds 1, 2, 3, each 43 rows | 0 |
| `rs_cpc_m_prototype_init_ablation_seed_rows.csv` | 99 | seeds 1, 2, 3, each 33 rows | 0 |

The 129 cache-tradeoff seed rows exclude zero-shot rows, which is expected for cache-size tradeoff analysis.

Method coverage:

- `zero_shot`: 3 seed rows, `shot_label=zero_shot`
- `tip_adapter`: 15 seed rows
- `proto_adapter`: 15 seed rows
- `rs_cpc`: 99 seed rows

Shot coverage:

- `zero_shot`: 3 rows
- 1-shot: 15 rows
- 2-shot: 21 rows
- 4-shot: 27 rows
- 8-shot: 33 rows
- 16-shot: 33 rows

RS-CPC M/prototype-init coverage follows the expected `M <= shot` structure:

| Shot | M values present | Prototype init values |
| ---: | --- | --- |
| 1 | 1 | `mean`, `medoid`, `random_group_mean` |
| 2 | 1, 2 | M1: `mean`, `medoid`, `random_group_mean`; M2: `medoid`, `random_group_mean` |
| 4 | 1, 2, 4 | M1: `mean`, `medoid`, `random_group_mean`; M2/M4: `medoid`, `random_group_mean` |
| 8 | 1, 2, 4, 8 | M1: `mean`, `medoid`, `random_group_mean`; M2/M4/M8: `medoid`, `random_group_mean` |
| 16 | 1, 2, 4, 8 | M1: `mean`, `medoid`, `random_group_mean`; M2/M4/M8: `medoid`, `random_group_mean` |

No abnormal seed, shot, method, or prototype-init coverage was found relative to this 132-run whitelist. `kmeans` is not present in this Day4 package, but it is not counted as missing against the verified Day4 whitelist.

## Main Accuracy Interpretation

Zero-shot AID + RemoteCLIP is strong:

- `zero_shot`: 86.87% mean top-1, 0.67% std.

Few-shot accuracy trends:

| Shot | Tip-Adapter | Proto-Adapter | Best RS-CPC | Best RS-CPC delta vs Tip |
| ---: | ---: | ---: | --- | ---: |
| 1 | 85.10 +/- 1.80 | 83.32 +/- 1.56 | M1 mean, 83.32 +/- 1.56 | -1.78 pp |
| 2 | 88.50 +/- 0.18 | 91.08 +/- 0.68 | M1 mean, 91.08 +/- 0.68 | +2.58 pp |
| 4 | 90.82 +/- 0.78 | 92.95 +/- 0.25 | M1 mean, 92.95 +/- 0.25 | +2.13 pp |
| 8 | 90.98 +/- 0.96 | 94.03 +/- 0.74 | M1 mean, 94.03 +/- 0.74 | +3.05 pp |
| 16 | 91.75 +/- 0.26 | 94.62 +/- 0.41 | M1 mean, 94.62 +/- 0.41 | +2.87 pp |

Interpretation:

- Tip-Adapter is best only at 1-shot.
- From 2-shot through 16-shot, Proto-Adapter and RS-CPC M1 mean tie for best mean accuracy.
- RS-CPC M1 mean is effectively the compact one-prototype setting and matches Proto-Adapter exactly in this package.
- AID accuracy rises quickly from 1-shot to 4-shot, then saturates near 94-95% by 8-shot and 16-shot.
- Larger M does not improve AID accuracy in this table package. On AID, compact class-level prototypes appear sufficient for RemoteCLIP features.

## Efficiency Interpretation

All included methods are training-free in this package:

- Mean trainable parameters: 0.
- Mean training time: 0 seconds.
- GPU memory field: not available in the lightweight summaries.

The efficiency trend is dominated by cache entries.

At 16-shot:

| Method variant | Cache entries | Inference time | Images/sec |
| --- | ---: | ---: | ---: |
| Proto-Adapter | 30 | 4.95 sec | 808.15 |
| Tip-Adapter | 480 | 40.79 sec | 98.09 |
| RS-CPC M1 mean | 30 | 4.95 sec | 808.27 |
| RS-CPC M2 random_group_mean | 60 | 7.24 sec | 552.94 |
| RS-CPC M4 random_group_mean | 120 | 11.89 sec | 337.03 |
| RS-CPC M8 random_group_mean | 240 | 21.10 sec | 189.79 |

At 16-shot, compared with Tip-Adapter:

- RS-CPC M1 uses 1/16 the cache entries and is about 8.2x faster by mean inference time.
- RS-CPC M2 uses 1/8 the cache entries and is about 5.6x faster.
- RS-CPC M4 uses 1/4 the cache entries and is about 3.4x faster.
- RS-CPC M8 uses 1/2 the cache entries and is about 1.9x faster.

Across shots, Tip-Adapter inference time grows with shot count and cache size:

- 1-shot: 30 entries, 5.25 sec, 761.85 images/sec.
- 16-shot: 480 entries, 40.79 sec, 98.09 images/sec.

RS-CPC scales with M instead of shot count:

- M1 stays near 5 sec because it keeps 30 entries.
- M2 stays near 7.2 sec because it keeps 60 entries.
- M4 stays near 11.9 sec because it keeps 120 entries.
- M8 stays near 21 sec because it keeps 240 entries.

This supports the intended efficiency story for a compact prototype cache, but these remain candidate analysis tables rather than final paper-facing results.

## Cache-Size Tradeoff

The key 16-shot tradeoff:

| Method variant | Cache entries | Compression ratio vs Tip cache | Mean top-1 | Inference time |
| --- | ---: | ---: | ---: | ---: |
| Tip-Adapter | 480 | 1x baseline | 91.75 | 40.79 sec |
| Proto-Adapter | 30 | 16x | 94.62 | 4.95 sec |
| RS-CPC M1 mean | 30 | 16x | 94.62 | 4.95 sec |
| RS-CPC M2 random_group_mean | 60 | 8x | 94.45 | 7.24 sec |
| RS-CPC M4 random_group_mean | 120 | 4x | 94.22 | 11.89 sec |
| RS-CPC M8 random_group_mean | 240 | 2x | 93.73 | 21.10 sec |

On AID + RemoteCLIP, the compact side of the tradeoff is strongest:

- M1 has the best accuracy and the smallest cache.
- M2/M4 keep accuracy close to M1 while still much smaller and faster than Tip-Adapter.
- M8 is still faster than Tip-Adapter but is less accurate than M1/M2/M4.
- Larger M does not recover a meaningful accuracy gain on AID in this package.

At 8-shot:

- Tip-Adapter: 240 entries, 90.98% top-1, 22.10 sec.
- RS-CPC M1 mean: 30 entries, 94.03% top-1, 5.08 sec.
- RS-CPC M4 random_group_mean: 120 entries, 92.33% top-1, 11.92 sec.
- RS-CPC M8: 240 entries, 90.65% top-1, about 21 sec.

This again indicates that increasing cache size beyond M1 is not helpful for AID under the current configuration.

## RS-CPC M / Prototype Init Ablation

Best RS-CPC variant by shot:

| Shot | Best variant | Mean top-1 | Std |
| ---: | --- | ---: | ---: |
| 1 | M1 mean | 83.32 | 1.56 |
| 2 | M1 mean | 91.08 | 0.68 |
| 4 | M1 mean | 92.95 | 0.25 |
| 8 | M1 mean | 94.03 | 0.74 |
| 16 | M1 mean | 94.62 | 0.41 |

Prototype-init observations:

- `mean` is only evaluated for M1 and is consistently best or tied for best.
- `random_group_mean` ties `mean` at M1 and is usually much stronger than `medoid` for M > 1.
- `medoid` is weak for AID when shot >= 2. At 16-shot, M2/M4 medoid are around 88.5%, while random-group mean variants are above 93.7%.
- The medoid behavior is a method-performance signal, not a coverage anomaly: all expected medoid rows are present and have complete seed coverage.

Representative 16-shot ablation:

| M | Init | Mean top-1 | Std | Entries | Compression ratio | Inference time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | mean | 94.62 | 0.41 | 30 | 16x | 4.95 sec |
| 1 | medoid | 91.73 | 0.75 | 30 | 16x | 4.96 sec |
| 1 | random_group_mean | 94.62 | 0.41 | 30 | 16x | 4.98 sec |
| 2 | medoid | 88.52 | 0.38 | 60 | 8x | 7.24 sec |
| 2 | random_group_mean | 94.45 | 0.44 | 60 | 8x | 7.24 sec |
| 4 | medoid | 88.58 | 0.60 | 120 | 4x | 11.88 sec |
| 4 | random_group_mean | 94.22 | 0.67 | 120 | 4x | 11.89 sec |
| 8 | medoid | 89.93 | 0.49 | 240 | 2x | 21.19 sec |
| 8 | random_group_mean | 93.73 | 0.33 | 240 | 2x | 21.10 sec |

## Lightweight Alignment With Day2 EuroSAT + RemoteCLIP

Day2 EuroSAT + RemoteCLIP was checked only through the existing table summary files in:

`results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z`

Day2 has the same high-level audit status:

- 132 preflight rows.
- 132 included rows.
- 0 excluded rows.
- `remote_server` only.
- `server_full` only.
- `paper_facing_status`: `not_marked_as_final_paper_tables`.
- All raw paper-result and paper-eligibility flags are false.

Accuracy alignment:

| Shot | EuroSAT best RS-CPC | AID best RS-CPC | AID - EuroSAT |
| ---: | --- | --- | ---: |
| 1 | M1 mean, 57.17 | M1 mean, 83.32 | +26.15 pp |
| 2 | M2 medoid, 67.80 | M1 mean, 91.08 | +23.29 pp |
| 4 | M2 random_group_mean, 70.72 | M1 mean, 92.95 | +22.23 pp |
| 8 | M4 random_group_mean, 77.06 | M1 mean, 94.03 | +16.98 pp |
| 16 | M4 random_group_mean, 82.52 | M1 mean, 94.62 | +12.10 pp |

Key differences:

- EuroSAT zero-shot RemoteCLIP is low in Day2 at 32.36%, while AID zero-shot RemoteCLIP is high at 86.87%.
- EuroSAT benefits from increasing M as shot count grows. Its best RS-CPC variants move from M1 at 1-shot to M4 at 8-shot and 16-shot.
- AID does not show the same need for multi-prototype expansion. M1 mean remains best from 1-shot through 16-shot.
- On EuroSAT, RS-CPC outperforms Tip-Adapter clearly at 4-shot, 8-shot, and 16-shot. On AID, Tip-Adapter is best only at 1-shot; Proto/RS-CPC M1 dominate from 2-shot onward.

Efficiency alignment:

- Both Day2 and Day4 show the expected cache-size scaling: Tip-Adapter grows with `C x K`, while RS-CPC grows with `C x M`.
- Direct wall-time comparison across datasets should be treated carefully because the test-set sizes and table contexts differ.
- Within each dataset, the efficiency pattern is stable: smaller cache means higher images/sec and lower inference time.

At 16-shot:

| Dataset | Variant | Entries | Inference time | Images/sec |
| --- | --- | ---: | ---: | ---: |
| EuroSAT | Tip-Adapter | 160 | 36.83 sec | 293.31 |
| EuroSAT | RS-CPC M4 random_group_mean | 40 | 12.29 sec | 878.88 |
| AID | Tip-Adapter | 480 | 40.79 sec | 98.09 |
| AID | RS-CPC M4 random_group_mean | 120 | 11.89 sec | 337.03 |

The shared message is that RS-CPC can preserve or improve accuracy while reducing cache size and inference cost, but the preferred M appears dataset-dependent.

## Day5 Conclusion

The Day4 AID + RemoteCLIP table package is internally complete for the 132-run verified whitelist:

- Expected runs: 132.
- Found runs with metadata and metrics: 132.
- Missing runs: 0.
- Included rows: 132.
- Excluded rows: 0.
- Only `remote_server` / `server_full` rows are included.
- No raw run is marked as a final paper result.

The current lightweight git package omits the two large seed-row files for confusion matrix and per-class accuracy. Their absence should be noted, but it does not block the Day5 audit of the lightweight package.

The AID trend is clear: RemoteCLIP is already strong, and a one-prototype compact cache is enough to match or beat the larger Tip-Adapter cache from 2-shot through 16-shot. Larger RS-CPC M values increase inference cost and usually reduce accuracy on AID. This contrasts with Day2 EuroSAT, where larger M becomes useful as shot count increases.

These findings should be treated as candidate verified analysis, not as final paper-facing claims.

## Day6 NWPU-RESISC45 + RemoteCLIP Artifact Preparation

Recommended next steps for Day6, without executing the Day6 full matrix:

1. Locate or prepare the NWPU-RESISC45 + RemoteCLIP table package directory and expected whitelist metadata.
2. Run only lightweight artifact checks: file presence, row counts, SHA256 verification for local lightweight files, included env/run-mode checks, and paper-flag checks.
3. Keep `paper_facing_status` as `not_marked_as_final_paper_tables` unless the user explicitly approves final paper inclusion.
4. Do not scan or modify `results/raw` during the summary audit unless the Day6 task explicitly provides a post-run preflight or table package requiring it.
5. If confusion-matrix or per-class seed rows are intentionally omitted from git, record that as a lightweight-package limitation rather than a failure.
6. Generate a Day6 artifact-preparation note under `results/summaries/`, separate from any future full matrix execution.
7. Do not execute the NWPU full matrix locally. If full runs are missing, prepare server-side commands or a checklist for user execution on the remote server.
