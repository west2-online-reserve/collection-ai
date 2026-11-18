import inspect
import json
from pathlib import Path
from typing import Any

from classes.util.util import to_dict
from const.enum import ConditionType, EffectType


class Dialogue:
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content


class Narration(Dialogue):
    def __init__(self, content: str):
        self.name = ""
        self.content = content


class Condition:
    def __init__(self, type: ConditionType, value: Any) -> None:
        self.type = type
        self.value = value


class Effect:
    def __init__(self, type: EffectType, value: Any) -> None:
        self.type = type
        self.value = value


def _get_caller_file() -> str:
    SKIP = {"character", "dialogue_helper"}
    for frame_info in inspect.stack():
        filename = Path(frame_info.filename).stem
        if filename not in SKIP:
            return filename
    return "unknown"


class Choice:
    def __init__(
        self,
        character: str,
        content: str,
        group_to: str,
        file_to: str = "",
    ):
        self.character = character
        self.content = content
        self.group_to = group_to
        self.file_to = file_to if file_to else _get_caller_file()
        self.effects: list[Effect] = []
        self.conditions: list[Condition] = []

    def add_condition(self, condition: Condition | list[Condition]):
        if isinstance(condition, Condition):
            self.conditions.append(condition)
        else:
            self.conditions.extend(condition)

        return self

    def add_effect(self, effect: Effect | list[Effect]):
        if isinstance(effect, Effect):
            self.effects.append(effect)
        else:
            self.effects.extend(effect)

        return self


class DialogueGroup:
    def __init__(
        self,
        id: str,
        dialogues: list[Dialogue | str],
        group_to: str,
        file_to: str = "",
    ):
        self.type = "dialogue_group"
        self.id = id
        self.dialogues = [
            Narration(dialogue) if isinstance(dialogue, str) else dialogue
            for dialogue in dialogues
        ]
        self.group_to = group_to
        self.file_to = (
            Path(inspect.stack()[1].filename).stem if file_to == "" else file_to
        )


class ChoiceGroup:
    def __init__(self, id: str, desc: str, choices: list[Choice]):
        self.type = "choice_group"
        self.id = id
        self.desc = desc
        self.choices = choices


class DialogueExporter:
    def __init__(
        self,
        groups: list[DialogueGroup | ChoiceGroup],
    ):
        output_dir: Path

        # 判断galgame/data文件夹在哪
        current = Path(__file__).resolve()
        for parent in current.parents:
            if parent.name == "galgame":
                output_dir = parent / "data" / "scene"
                break
        else:
            output_dir = Path("galgame/data/scene")

        file_name = Path(inspect.stack()[1].filename).stem

        output_path = Path(output_dir) / f"{file_name}.json"

        # 一键创建目录
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 数据导出
        data: dict[str, dict[str, Any]] = {}
        for group in groups:
            group_dict = to_dict(group)
            group_dict.pop("id", None)
            data[group.id] = group_dict

        # 生成json文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # print(f"已生成 {file_name}.json 于 {output_path}")
