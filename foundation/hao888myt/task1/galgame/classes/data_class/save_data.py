from dataclasses import dataclass, field
from typing import Any

from const.enum import EffectType


@dataclass
class SaveData:
    current_file: str = "start"
    current_group: str = "group_1"
    current_index: int = 0
    characters: dict[str, dict[str, Any]] = field(default_factory=dict[str, Any])

    def next_dialogue(self):
        self.current_index += 1

    def to_group(self, id: str):
        self.current_group = id
        self.current_index = 0

    def to_file(self, file_to: str):
        self.current_file = file_to

    def reset_current_index(self):
        self.current_index = 0

    def get_current_file(self):
        return self.current_file

    def get_current_index(self):
        return self.current_index

    def get_current_group(self):
        return self.current_group

    @classmethod
    def trans_from_dict(cls, data: dict[str, Any]) -> "SaveData":
        return cls(**data)

    def trans_to_dict(self) -> dict[str, Any]:
        return {
            "current_file": self.current_file,
            "current_group": self.current_group,
            "current_index": self.current_index,
            "characters": self.characters,
        }

    def apply_effect(self, choice: dict[str, Any]):
        character = choice["character"]

        for effect in choice.get("effects", []):
            type = effect["type"]
            value = effect["value"]

            match type:
                case EffectType.MODIFY_AFFINITY.value:
                    if character in self.characters:
                        self.characters[character]["affinity"] += value
                case _:
                    print("完全没有效果喵")
