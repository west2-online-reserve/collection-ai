import json
from pathlib import Path
from typing import Dict, Any

SAVE_DIRECTORY: Path = Path(__file__).parent / "saves"
NORMAL_SAVE_FILE: Path = SAVE_DIRECTORY / "save.json"
AUTO_SAVE_FILE: Path = SAVE_DIRECTORY / "autosave.json"


def save_game(data: Dict[str, Any], auto: bool = False) -> None:
    """将游戏写入普通或即时存档"""
    SAVE_DIRECTORY.mkdir(exist_ok=True)

    save_file: Path
    if auto:
        save_file = AUTO_SAVE_FILE
    else:
        save_file = NORMAL_SAVE_FILE

    with save_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    if not auto:
        print("存档成功！")


def load_game(auto: bool = False) -> Dict[str, Any]:
    """读取普通存档或即时存档。"""
    save_file: Path
    if auto:
        save_file = AUTO_SAVE_FILE
    else:
        save_file = NORMAL_SAVE_FILE

    if not save_file.exists():
        print("没有找到对应的存档。")
        return {}

    try:
        with save_file.open("r", encoding="utf-8") as file:
            data: Dict[str, Any] = json.load(file)
        return data
    except (json.JSONDecodeError, OSError):
        print("存档损坏或无法读取。")
        return {}
