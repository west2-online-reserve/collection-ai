from typing import Dict, Any
import json

SAVE_FILE = "save_data.json"

def save_progress(data: Dict[str, Any]) -> None:
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_progress() -> Dict[str, Any]:
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
