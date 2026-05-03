from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


Config = dict[str, Any]


class ConfigError(ValueError):
    """Raised when a config is missing required structure."""


def load_yaml_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    return data


def deep_merge(base: Config, override: Config) -> Config:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_by_dotted_key(config: Config, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        node = current.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Cannot set nested key under non-mapping: {dotted_key}")
        current = node
    current[parts[-1]] = value


def parse_override_value(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def apply_cli_overrides(config: Config, overrides: Iterable[str] | None) -> Config:
    updated = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ConfigError(f"Override must use key=value format: {override}")
        key, raw_value = override.split("=", 1)
        set_by_dotted_key(updated, key, parse_override_value(raw_value))
    return updated


def load_configs(paths: Iterable[str | Path], overrides: Iterable[str] | None = None) -> Config:
    config: Config = {}
    for path in paths:
        config = deep_merge(config, load_yaml_config(path))
    return apply_cli_overrides(config, overrides)


def validate_required_fields(config: Config, required_fields: Iterable[str]) -> None:
    missing: list[str] = []
    for dotted_key in required_fields:
        current: Any = config
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                missing.append(dotted_key)
                break
            current = current[part]
    if missing:
        raise ConfigError(f"Missing required config fields: {', '.join(missing)}")


def save_config_snapshot(config: Config, run_dir: str | Path, filename: str = "config.yaml") -> Path:
    destination = Path(run_dir) / filename
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite config snapshot: {destination}")
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    return destination
