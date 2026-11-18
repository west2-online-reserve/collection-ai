from enum import Enum
from pathlib import Path
from typing import Any


# Pylance我错了喵，泥不要在用类型不确定肘击我了
def to_dict(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]  # type: ignore
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}  # type: ignore
    if hasattr(obj, "__dict__"):
        return {key: to_dict(value) for key, value in obj.__dict__.items()}
    return obj


def get_galgame_path():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "galgame":
            path = parent
            break
    else:
        path = Path("galgame")

    return path
