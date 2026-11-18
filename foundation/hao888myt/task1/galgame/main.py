import importlib
import inspect
import json
from msvcrt import getch
from pathlib import Path
from typing import Any

from classes.data_class.save_data import SaveData
from classes.loader.file_loader import FileLoader
from classes.util.dialogue_printer import DialoguePrinter
from classes.util.util import get_galgame_path, to_dict
from const.enum import ConditionType
from const.path import SAVE_PATH, SCENE_PATH


def run_all_generators():
    for file in Path("scene").glob("*.py"):
        if file.name.startswith("__"):
            continue
        name = file.stem
        module = importlib.import_module(f"scene.{name}")
        if hasattr(module, "generator"):
            module.generator()
        else:
            print(f"{name}.py没有generator()函数")


def instantiate_all() -> dict[str, Any]:
    folder: str = "classes/data_class"
    results: dict[str, Any] = {}

    py_file = Path(f"{folder}/character_data.py")

    module_name = py_file.stem

    # 动态导入
    module = importlib.import_module(f"{folder.replace("/", ".")}.{module_name}")

    # 找模块里的类
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # 只处理本模块定义的类（排除导入的类）
        if obj.__module__ == module.__name__ and name != "CharacterData":
            instance = obj()

            character_dict = to_dict(instance)
            character_dict.pop("id", None)

            results[instance.id] = character_dict  # type: ignore
            print(f"已实例化 {module_name}.{name}")

    save_data = SaveData()
    save_data.characters = to_dict(results)

    return to_dict(save_data)


def create_new_game():
    output_path = get_galgame_path() / SAVE_PATH / "data.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(instantiate_all(), f, ensure_ascii=False, indent=2)


def load_game() -> tuple[SaveData, dict[str, Any]]:
    save = FileLoader(SAVE_PATH, "data").get_json()
    save_data = SaveData.trans_from_dict(save)

    scene = FileLoader(SCENE_PATH, save_data.current_file).get_json()
    return save_data, scene


def save_game(save_data: SaveData):
    save = save_data.trans_to_dict()

    output_path = get_galgame_path() / SAVE_PATH / "data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)


def check_choice(choice: dict[str, Any], characters: dict[str, Any]) -> bool:
    character = choice["character"]

    for condition in choice.get("conditions", []):
        match condition["type"]:
            case ConditionType.REQUIRED_AFFINITY.value:
                affinity = characters.get(character, {}).get("affinity", 0)
                if affinity < condition["value"]:
                    return False
            case _:
                pass
    return True


def game_loop():
    save_data, scene = load_game()

    while True:
        group = scene.get(save_data.get_current_group())
        if not group:
            break

        # 对话组
        if group["type"] == "dialogue_group":
            dialogues = group["dialogues"]

            if save_data.get_current_index() >= len(dialogues):
                save_data.to_group(group["group_to"])

                if (
                    group.get("file_to")
                    and group["file_to"] != save_data.get_current_file()
                ):
                    save_data.to_file(group["file_to"])
                    scene = FileLoader(
                        SCENE_PATH, save_data.get_current_file()
                    ).get_json()
                continue

            line = dialogues[save_data.get_current_index()]
            DialoguePrinter().print_line(line)
            getch()
            save_data.next_dialogue()

        # 选项组
        elif group["type"] == "choice_group":
            print(f"\n{group['desc']}")

            choices = [
                choice
                for choice in group["choices"]
                if check_choice(choice, save_data.characters)
            ]

            if not choices:
                print("没有可用选项")
                break

            for i, choice in enumerate(choices, 1):
                print(f"  {i}. {choice['content']}")

            while True:
                raw = input("> ").strip()
                if raw.isdigit() and 1 <= int(raw) <= len(choices):
                    index = int(raw) - 1
                    break
                print("无效输入，请重新选择")

            choice = choices[index]

            save_data.apply_effect(choice)

            save_data.to_group(choice["group_to"])

            if (
                choice.get("file_to")
                and choice["file_to"] != save_data.get_current_file()
            ):
                save_data.to_file(choice["file_to"])
                scene = FileLoader(SCENE_PATH, save_data.get_current_file()).get_json()

        save_game(save_data)


if __name__ == "__main__":
    run_all_generators()

    if not any(SAVE_PATH.iterdir()):
        print("没有玩家存档，正在创建中")
        create_new_game()

    game_loop()
