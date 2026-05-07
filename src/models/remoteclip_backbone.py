from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.base_backbone import BaseBackbone, BackboneConfig, BackboneUnavailableError


class RemoteClipBackbone(BaseBackbone):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        self.model: Any | None = None

    def load_model(self) -> "RemoteClipBackbone":
        if self.config.dry_run:
            self.is_loaded = True
            self.is_eval = True
            return self
        if not self.config.weights:
            raise BackboneUnavailableError(
                "RemoteCLIP requires an explicit local weights path; automatic downloads are disabled."
            )
        weights_path = Path(self.config.weights)
        if not weights_path.exists():
            raise BackboneUnavailableError(f"RemoteCLIP weights do not exist: {weights_path}")
        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise BackboneUnavailableError(
                "RemoteCLIP real loading requires torch and open_clip installed in the server environment."
            ) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise BackboneUnavailableError(f"CUDA device requested but unavailable: {self.config.device}")

        model = open_clip.create_model("ViT-B-32", pretrained=None, device=self.config.device)
        checkpoint = torch.load(weights_path, map_location=self.config.device)
        state_dict = checkpoint_state_dict(checkpoint)
        load_result = model.load_state_dict(strip_state_dict_prefixes(state_dict), strict=False)
        model.eval()
        self.model = model
        self.is_loaded = True
        self.is_eval = True
        self.load_result = {
            "missing_keys": list(getattr(load_result, "missing_keys", [])),
            "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        }
        return self


def checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise BackboneUnavailableError("RemoteCLIP checkpoint is not a state dict or checkpoint mapping.")


def strip_state_dict_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned
