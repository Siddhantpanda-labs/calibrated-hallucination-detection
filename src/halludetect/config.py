"""
Configuration loader and fallback utilities.
"""

from __future__ import annotations

import os
from typing import Any, Iterable
import pandas as pd


def safe_tqdm(iterable: Iterable, **kwargs) -> Iterable:
    """Safe wrapper for tqdm that yields items directly if tqdm is missing."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def df_to_markdown(df: pd.DataFrame) -> str:
    """Formats dataframe as markdown table without requiring tabulate."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def load_config(config_path: str = "configs/default.yaml") -> dict[str, Any]:
    """
    Loads YAML or JSON configuration safely.
    Falls back gracefully if PyYAML is not installed.
    """
    if not os.path.exists(config_path):
        return {}

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass

    # Simple fallback parser for basic YAML key: value / nesting
    cfg: dict[str, Any] = {}
    current_section = None
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith("#"):
                continue
            if not line.startswith(" ") and line_strip.endswith(":"):
                current_section = line_strip[:-1]
                cfg[current_section] = {}
            elif ":" in line_strip:
                k, v = line_strip.split(":", 1)
                k = k.strip()
                v = v.split("#")[0].strip().strip("\"'")
                if v.lower() == "true":
                    val: Any = True
                elif v.lower() == "false":
                    val = False
                else:
                    try:
                        val = int(v) if "." not in v else float(v)
                    except ValueError:
                        val = v
                if current_section:
                    cfg[current_section][k] = val
                else:
                    cfg[k] = val
    return cfg
