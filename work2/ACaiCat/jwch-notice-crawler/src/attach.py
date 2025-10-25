from dataclasses import dataclass
from typing import Any

from constants import BASE_URL


@dataclass
class Attach:
    name: str
    path: str
    download_times: int

    @property
    def url(self) -> str:
        return BASE_URL + self.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "download_times": self.download_times,
            "url": self.url
        }
