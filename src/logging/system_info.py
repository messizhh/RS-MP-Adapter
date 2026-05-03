from __future__ import annotations

import platform
import socket
import subprocess
import sys

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def git_commit_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def get_system_info(device: str = "cpu") -> dict[str, object]:
    cuda_version = "not_available"
    pytorch_version = "not_installed"
    gpu_name = "CPU"
    if torch is not None:
        pytorch_version = torch.__version__
        cuda_version = torch.version.cuda or "not_available"
        if device.startswith("cuda") and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    return {
        "git_commit": git_commit_hash(),
        "python_version": sys.version.replace("\n", " "),
        "pytorch_version": pytorch_version,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "host_name": socket.gethostname(),
        "platform": platform.platform(),
    }
