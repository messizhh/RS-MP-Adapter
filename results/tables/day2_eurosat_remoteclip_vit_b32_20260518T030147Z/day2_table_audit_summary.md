# Day 2 Table Audit

Source preflight summary: `outputs/preflight/post_run_result/eurosat_remoteclip_vit_b32/full_matrix_20260518T022057Z/post_run_preflight_summary.tsv`
Output directory: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z`

This package is a verified analysis table generated from the full-matrix whitelist and post-run preflight.
It preserves the raw `is_paper_result` and `eligible_for_paper_tables` flags and does not mark these rows as final paper-facing results.

## Counts

- Preflight rows: 132
- Included rows: 132
- Excluded rows: 0
- Included execution envs: remote_server
- Included run modes: server_full

## Raw Paper Flags

- metrics_is_paper_result_true: 0
- metrics_is_paper_result_false: 132
- metadata_is_paper_result_true: 0
- metadata_is_paper_result_false: 132
- metrics_eligible_for_paper_tables_true: 0
- metrics_eligible_for_paper_tables_false: 132
- metadata_eligible_for_paper_tables_true: 0
- metadata_eligible_for_paper_tables_false: 132

## Per-Class and Confusion Matrix

- Per-class accuracy: available
- Confusion matrix: available

## Exclusion Reasons

- None

## Outputs

- inclusion_registry_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/inclusion_registry.csv`
- inclusion_registry_json: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/inclusion_registry.json`
- main_accuracy_seed_rows_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/main_accuracy_seed_rows.csv`
- main_accuracy_summary_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/main_accuracy_summary.csv`
- efficiency_seed_rows_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/efficiency_seed_rows.csv`
- efficiency_summary_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/efficiency_summary.csv`
- cache_tradeoff_seed_rows_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/cache_tradeoff_seed_rows.csv`
- cache_tradeoff_summary_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/cache_tradeoff_summary.csv`
- rs_cpc_ablation_seed_rows_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/rs_cpc_m_prototype_init_ablation_seed_rows.csv`
- rs_cpc_ablation_summary_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/rs_cpc_m_prototype_init_ablation_summary.csv`
- per_class_accuracy_seed_rows_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/per_class_accuracy_seed_rows.csv`
- confusion_matrix_seed_rows_csv: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/confusion_matrix_seed_rows.csv`
- audit_summary_json: `results/tables/day2_eurosat_remoteclip_vit_b32_20260518T030147Z/day2_table_audit_summary.json`
