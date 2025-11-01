from enum import Enum


class GameOption(Enum):
    TALK = 1
    GIFT = 2
    STATUS = 3
    EXIT = 4
    INVALID = 0

    @classmethod
    def prase(cls, input_value: str):
        try:
            return cls(int(input_value))
        except (ValueError, TypeError):
            return cls.INVALID

