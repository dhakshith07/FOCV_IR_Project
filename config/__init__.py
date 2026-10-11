"""
config/ — Configuration loader module.
Provides a singleton AppConfig object loaded from config.yaml.
"""
from config.settings import AppConfig, load_config

__all__ = ["AppConfig", "load_config"]
