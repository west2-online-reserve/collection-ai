import json
from pathlib import Path
from typing import Any

from classes.util.util import get_galgame_path


class FileLoader:
    def __init__(self, path: Path, file_name: str):

        self.file_path = get_galgame_path() / path / f"{file_name}.json"

    def load(self) -> dict[str, Any]:
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_groups(self) -> list[dict[str, Any]]:
        data = self.load()
        return data.get("groups", [])

    def get_group(self, group_id: str) -> dict[str, Any] | None:
        for group in self.get_groups():
            if group.get("id") == group_id:
                return group
        return None

    def get_json(self):
        return self.load()
