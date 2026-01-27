"""
Configuration Management

Single source of truth for all system parameters.
Loads from YAML config file with validation and defaults.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional
import yaml


def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep-merge override into base (in place)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file. Defaults to config/default.yaml
    
    Returns:
        Config dictionary with all parameters
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required keys are missing
    """
    # Default to config/default.yaml relative to project root
    project_root = Path(__file__).parent.parent.parent
    default_path = project_root / "config" / "default.yaml"
    if config_path is None:
        config_path = default_path
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty or invalid: {config_path}")
    
    # Merge with defaults if provided config is partial
    required_keys = [
        "universe", "liquidity", "technicals", "outputs", "runtime"
    ]
    missing = [k for k in required_keys if k not in config]
    if missing and config_path != default_path:
        with open(default_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}
        config = _merge_dicts(base_config, config)
        missing = [k for k in required_keys if k not in config]

    # Validate required top-level keys
    required_keys = [
        "universe", "liquidity", "technicals", "outputs", "runtime"
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    
    # Apply environment variable overrides (optional)
    # Format: MOMENTUM_<SECTION>_<KEY> (uppercase, underscores)
    # Example: MOMENTUM_MOVERS_ENABLED=true
    for key, value in os.environ.items():
        if key.startswith("MOMENTUM_") and "_" in key[9:]:
            parts = key[9:].lower().split("_")
            if len(parts) >= 2:
                section = parts[0]
                subkey = "_".join(parts[1:])
                if section in config and isinstance(config[section], dict):
                    # Type conversion for common types
                    if isinstance(config[section].get(subkey), bool):
                        value = value.lower() in ("true", "1", "yes", "on")
                    elif isinstance(config[section].get(subkey), (int, float)):
                        try:
                            value = float(value) if "." in value else int(value)
                        except ValueError:
                            continue
                    config[section][subkey] = value
    
    return config


def get_config_value(config: dict, *keys: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Example:
        get_config_value(config, "movers", "enabled") -> config["movers"]["enabled"]
    
    Args:
        config: Config dictionary
        keys: Nested keys to traverse
        default: Default value if key path doesn't exist
    
    Returns:
        Config value or default
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

