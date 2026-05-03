from __future__ import annotations

from src.models.base_backbone import BaseBackbone, BackboneConfig


class ClipBackbone(BaseBackbone):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
