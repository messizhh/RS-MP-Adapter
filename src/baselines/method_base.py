from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.eval.evaluator import evaluate_logits


class MethodBase:
    method_name = "method_base"

    def fit(self, support_features, support_labels, val_features=None, val_labels=None):
        raise NotImplementedError

    def predict_logits(self, image_features):
        raise NotImplementedError

    def evaluate(self, image_features, labels) -> dict[str, Any]:
        return evaluate_logits(self.predict_logits(image_features), labels, class_names=getattr(self, "class_names", None))

    def state_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite method state: {destination}")
        with destination.open("wb") as handle:
            pickle.dump({"class": self.__class__.__name__, "state": self.state_dict()}, handle)
        return destination

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        instance = cls()
        instance.load_state_dict(payload["state"])
        return instance
