#!/usr/bin/env bash
set -euo pipefail

# Template only. Non-experimental server preflight.
# The user must manually edit/export TODO variables and run this on the remote server.
# This script must not train, evaluate, extract full features, download data, or download weights.

: "${DATASET:=nwpu_resisc45}"
: "${DATASET_CONFIG:=configs/datasets/nwpu_resisc45.yaml}"
: "${DATASET_ROOT:=TODO_DATASET_ROOT}"
: "${FEATURE_ROOT:=TODO_FEATURE_ROOT}"
: "${RESULT_ROOT:=TODO_RESULT_ROOT}"
: "${LOG_ROOT:=TODO_LOG_ROOT}"
: "${RUN_DATASET_LAYOUT_CHECK:=0}"

if [[ -z "${CHECKPOINT_ROOT:-}" && -n "${WEIGHT_ROOT:-}" ]]; then
  echo "warning=WEIGHT_ROOT is deprecated; use CHECKPOINT_ROOT instead." >&2
  CHECKPOINT_ROOT="${WEIGHT_ROOT}"
fi
: "${CHECKPOINT_ROOT:=TODO_CHECKPOINT_ROOT}"

echo "server_preflight=non_experimental"
echo "dataset=${DATASET}"

if [[ "${DATASET_ROOT}" == TODO_* || "${FEATURE_ROOT}" == TODO_* || "${CHECKPOINT_ROOT}" == TODO_* || "${RESULT_ROOT}" == TODO_* || "${LOG_ROOT}" == TODO_* ]]; then
  echo "error=Fill DATASET_ROOT, FEATURE_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, and LOG_ROOT before running on the server." >&2
  exit 2
fi

python - <<'PY'
import json
import shutil
import sys

report = {
    "python": sys.version.split()[0],
    "torch_available": False,
    "cuda_available": False,
    "cuda_device_count": 0,
    "gpu_names": [],
}
try:
    import torch
    report["torch_available"] = True
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count()) if report["cuda_available"] else 0
    report["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(report["cuda_device_count"])]
except Exception as exc:
    report["torch_error"] = str(exc)
report["nvidia_smi_available"] = shutil.which("nvidia-smi") is not None
print("environment_report=" + json.dumps(report, sort_keys=True))
if not report["torch_available"] or not report["cuda_available"] or report["cuda_device_count"] < 1:
    raise SystemExit(2)
PY

test -d "${DATASET_ROOT}" || { echo "error=DATASET_ROOT does not exist or is not a directory: ${DATASET_ROOT}" >&2; exit 2; }
test -d "${FEATURE_ROOT}" || { echo "error=FEATURE_ROOT does not exist or is not a directory: ${FEATURE_ROOT}" >&2; exit 2; }
test -d "${CHECKPOINT_ROOT}" || { echo "error=CHECKPOINT_ROOT does not exist or is not a directory: ${CHECKPOINT_ROOT}" >&2; exit 2; }
mkdir -p "${RESULT_ROOT}/raw" "${RESULT_ROOT}/tables" "${RESULT_ROOT}/figures" "${RESULT_ROOT}/summaries/preflight" "${LOG_ROOT}"
test -w "${RESULT_ROOT}" || { echo "error=RESULT_ROOT is not writable: ${RESULT_ROOT}" >&2; exit 2; }
test -w "${LOG_ROOT}" || { echo "error=LOG_ROOT is not writable: ${LOG_ROOT}" >&2; exit 2; }

if [[ "${RUN_DATASET_LAYOUT_CHECK}" == "1" ]]; then
  python scripts/check_dataset_layout.py \
    --config "${DATASET_CONFIG}" \
    --dataset "${DATASET}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${RESULT_ROOT}/summaries/preflight" \
    --execution-env remote_server \
    --run-mode server_benchmark
else
  echo "dataset_layout_check=skipped"
  echo "set RUN_DATASET_LAYOUT_CHECK=1 to run the read-only dataset layout preflight"
fi

echo "server_preflight_status=passed_non_experimental_checks"
