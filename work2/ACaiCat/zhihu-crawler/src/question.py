from dataclasses import dataclass
from typing import Any


@dataclass
class Question:
    title: str
    body: str
    url: str
    answers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "url": self.url,
            "answers": list[str]
        }

    def __str__(self) -> str:
        return str(self.to_dict())
