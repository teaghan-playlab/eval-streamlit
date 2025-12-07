"""Configuration loading helpers for the evaluation app."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .evaluation import EvaluatorConfig


def load_config(config_path: Path) -> EvaluatorConfig:
    """Load evaluator configuration from a JSON file.

    This mirrors the original `load_config` logic from the CLI script, but is
    placed in a reusable module so UI code doesn't need to know about JSON
    parsing details.
    
    Expects `categories` key in the config.
    """

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data: Dict[str, Any] = json.load(f)

    config = EvaluatorConfig(**config_data)
    # Set config_file path for tracking / metadata
    config.config_file = str(config_path)
    return config


def fields_to_categories(fields: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert a config ``fields`` mapping into a list of categories.

    Each category is a simple dict:
    ``{"label": str, "key": str, "description": str}``.
    """
    categories: List[Dict[str, str]] = []
    for key, spec in fields.items():
        description = spec.get("description", "")
        # Simple human-readable label from key, e.g. "gesi_alignment" -> "Gesi alignment"
        label = key.replace("_", " ").capitalize()
        categories.append({"label": label, "key": key, "description": description})
    return categories


def categories_to_fields(categories: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """Convert a list of categories into the EvaluatorConfig.categories format.

    Converts from UI format [{"label": "...", "key": "...", "description": "..."}]
    to config format {"key": {"description": "..."}}.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for cat in categories:
        key = cat["key"]
        description = cat.get("description", "")
        result[key] = {"description": description}
    return result


def slugify_label(label: str, existing_keys: List[str]) -> str:
    """Create a machine-friendly key from a human label, ensuring uniqueness.

    - Lowercases
    - Converts whitespace to underscores
    - Strips characters that are not ``[a-z0-9_]``
    - De-duplicates by appending ``_2``, ``_3``, ... when necessary
    """

    base = label.strip().lower()
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"[^a-z0-9_]", "", base)
    if not base:
        base = "category"

    key = base
    counter = 2
    while key in existing_keys:
        key = f"{base}_{counter}"
        counter += 1
    return key
