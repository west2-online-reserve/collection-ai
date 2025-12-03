import json
import os
from typing import Dict, Any, Optional
SAVE_DIR = "saves"

def save(data: Dict[str, Any], is_quick: bool = False) -> None:
    # 文件路径
    if is_quick: # 快速存档
        file_path = os.path.join(SAVE_DIR, "quick.json")
    else: # 常规存档
        slot = input('存储的存档编号:（数字）')
        file_path = os.path.join(SAVE_DIR, f"{slot}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{'即时' if is_quick else ''}存档成功")


def load(is_quick: bool = False) -> Optional[Dict[str, Any]]:
    # 文件路径
    if is_quick:
        file_path = os.path.join(SAVE_DIR, "quick.json")
    else:
        slot = input('读取的存档编号:（数字）')
        file_path = os.path.join(SAVE_DIR, f"{slot}.json")

    # 检查存档文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：存档文件不存在 → {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)