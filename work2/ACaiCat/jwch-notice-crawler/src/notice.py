import datetime
from dataclasses import dataclass

from attach import Attach
from constants import BASE_URL


@dataclass
class Notice:
    author: str
    title: str
    date: datetime.date
    path: str
    body: str
    attaches: list[Attach]

    @property
    def url(self) -> str:
        return BASE_URL + self.path


