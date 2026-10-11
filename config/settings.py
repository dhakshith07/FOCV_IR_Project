"""
config/settings.py
Loads and exposes the global YAML configuration as a typed dataclass-like object.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


# ── Project root (one level above this file) ─────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_FILE: Path = PROJECT_ROOT / "config.yaml"


# ── Raw dict loaded from YAML ────────────────────────────────────────────────
class _DotDict(dict):
    """A dict subclass that allows attribute-style access: cfg.app.name"""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict):
                return _DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Safe nested access: cfg.get_nested('vision', 'yolo', 'confidence_threshold')"""
        current = self
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


class AppConfig:
    """
    Singleton configuration object.
    Access anywhere via:
        from config.settings import get_config
        cfg = get_config()
        cfg.app.name  →  "Context-Aware Workspace Organizer"
    """

    _instance: Optional["AppConfig"] = None
    _raw: Optional[_DotDict] = None

    def __new__(cls) -> "AppConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Path = CONFIG_FILE) -> None:
        """Load (or reload) configuration from a YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._raw = _DotDict(raw)
        logger.debug(f"Configuration loaded from: {config_path}")

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        if self._raw is None:
            raise RuntimeError("AppConfig not loaded. Call load() first.")
        return getattr(self._raw, key)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Safe nested key access."""
        if self._raw is None:
            return default
        return self._raw.get_nested(*keys, default=default)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def __repr__(self) -> str:
        return f"<AppConfig loaded={self._raw is not None}>"


# ── Module-level singleton ─────────────────────────────────────────────────
_config_instance = AppConfig()


def load_config(config_path: Path = CONFIG_FILE) -> AppConfig:
    """Load config and return the singleton instance."""
    _config_instance.load(config_path)
    return _config_instance


def get_config() -> AppConfig:
    """Return the loaded config singleton. Loads it if not yet loaded."""
    if _config_instance._raw is None:
        _config_instance.load()
    return _config_instance
