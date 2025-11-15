import yaml
from pathlib import Path
from typing import Any

__all__ = ["load_yaml"]

def load_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
