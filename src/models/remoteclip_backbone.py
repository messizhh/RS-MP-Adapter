from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.models.base_backbone import BaseBackbone, BackboneConfig, BackboneUnavailableError


class RemoteClipBackbone(BaseBackbone):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        self.model: Any | None = None
        self.load_metadata: dict[str, Any] = {
            "checkpoint_loaded": False,
            "checkpoint_num_tensors": 0,
            "checkpoint_load_mode": "not_attempted",
            "missing_keys_count": 0,
            "unexpected_keys_count": 0,
            "missing_keys_sample": [],
            "unexpected_keys_sample": [],
            "model_class": None,
            "open_clip_initial_pretrained": None,
            "open_clip_initialization_warning_expected": False,
            "checkpoint_load_happened_after_model_init": False,
            "final_weights_loaded_from_checkpoint": False,
            "final_weight_source": None,
            "final_checkpoint_load_status": "not_attempted",
        }

    def load_model(self) -> "RemoteClipBackbone":
        if self.config.dry_run:
            self.is_loaded = True
            self.is_eval = True
            self.load_metadata = {
                **self.load_metadata,
                "checkpoint_load_mode": "dry_run",
                "model_class": self.__class__.__name__,
            }
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

        logging.warning(
            "RemoteCLIP: open_clip model is initialized before loading local checkpoint; "
            'the open_clip "No pretrained weights loaded" warning is expected when pretrained=None.'
        )
        model = open_clip.create_model("ViT-B-32", pretrained=None, device=self.config.device)
        logging.warning("RemoteCLIP: checkpoint load result follows after local checkpoint is read.")
        checkpoint = torch.load(weights_path, map_location=self.config.device)
        state_dict, load_mode = checkpoint_state_dict(checkpoint)
        cleaned_state_dict = strip_state_dict_prefixes(state_dict)
        checkpoint_num_tensors = count_tensor_like_values(cleaned_state_dict)
        base_load_metadata = {
            **self.load_metadata,
            "open_clip_initial_pretrained": None,
            "open_clip_initialization_warning_expected": True,
            "checkpoint_load_happened_after_model_init": True,
            "final_weights_loaded_from_checkpoint": False,
            "final_weight_source": "local_checkpoint",
            "model_class": model.__class__.__name__,
        }
        if checkpoint_num_tensors == 0:
            raise_with_metadata(
                "RemoteCLIP checkpoint contains zero tensor-like entries.",
                {
                    **base_load_metadata,
                    "checkpoint_load_mode": load_mode,
                    "checkpoint_num_tensors": 0,
                    "final_checkpoint_load_status": "not_loaded_empty_checkpoint",
                },
            )
        load_result = model.load_state_dict(cleaned_state_dict, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        checkpoint_loaded = checkpoint_num_tensors > 0 and not missing_keys and not unexpected_keys
        final_checkpoint_load_status = (
            "loaded_strictly_matching_keys" if checkpoint_loaded else "invalid_key_mismatch"
        )
        load_metadata = {
            **base_load_metadata,
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_num_tensors": checkpoint_num_tensors,
            "checkpoint_load_mode": load_mode,
            "missing_keys_count": len(missing_keys),
            "unexpected_keys_count": len(unexpected_keys),
            "missing_keys_sample": missing_keys[:10],
            "unexpected_keys_sample": unexpected_keys[:10],
            "final_weights_loaded_from_checkpoint": checkpoint_loaded,
            "final_checkpoint_load_status": final_checkpoint_load_status,
        }
        if not checkpoint_loaded:
            raise_with_metadata(
                "RemoteCLIP checkpoint did not strictly match the model keys; refusing to continue with partial or ambiguous weights.",
                load_metadata,
            )
        logging.warning(
            "RemoteCLIP: checkpoint load status=%s tensors=%s missing_keys=%s unexpected_keys=%s",
            final_checkpoint_load_status,
            checkpoint_num_tensors,
            len(missing_keys),
            len(unexpected_keys),
        )
        model.eval()
        self.model = model
        self.is_loaded = True
        self.is_eval = True
        self.load_metadata = load_metadata
        return self

    def encode_image_preflight(self, image_tensor: Any) -> Any:
        self._require_loaded()
        if self.model is None:
            raise BackboneUnavailableError("RemoteCLIP model is not available for image feature preflight.")
        try:
            import torch
        except ImportError as exc:
            raise BackboneUnavailableError("RemoteCLIP image feature preflight requires torch.") from exc
        with torch.no_grad():
            return self.model.encode_image(image_tensor.to(self.device))

    def encode_text_preflight(self, prompts: list[str]) -> Any:
        self._require_loaded()
        if self.model is None:
            raise BackboneUnavailableError("RemoteCLIP model is not available for text feature extraction.")
        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise BackboneUnavailableError("RemoteCLIP text feature extraction requires torch and open_clip.") from exc
        tokenizer = getattr(open_clip, "tokenize", None)
        if tokenizer is None and hasattr(open_clip, "get_tokenizer"):
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
        if tokenizer is None:
            raise BackboneUnavailableError("open_clip tokenizer is unavailable for RemoteCLIP text extraction.")
        tokens = tokenizer(prompts).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens)


def checkpoint_state_dict(checkpoint: Any) -> tuple[dict[str, Any], str]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key
        return checkpoint, "direct_state_dict"
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


def count_tensor_like_values(state_dict: dict[str, Any]) -> int:
    count = 0
    for value in state_dict.values():
        if hasattr(value, "shape") or hasattr(value, "numel"):
            count += 1
    return count


def raise_with_metadata(message: str, metadata: dict[str, Any]) -> None:
    error = BackboneUnavailableError(message)
    error.load_metadata = metadata
    raise error
