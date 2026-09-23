"""Configuration loading and early validation for reproducible experiments."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate Phase 1 invariants."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = ("simulation", "network", "channel", "power", "traffic", "digital_twin", "reward")
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    network = config["network"]
    if network["num_ues"] <= 0 or network["total_bandwidth_hz"] <= 0:
        raise ValueError("num_ues and total_bandwidth_hz must be positive")
    if network["num_resource_blocks"] <= 0:
        raise ValueError("num_resource_blocks must be positive")
    if config["power"]["min_dbm"] > config["power"]["max_dbm"]:
        raise ValueError("min_dbm cannot exceed max_dbm")
    return config
