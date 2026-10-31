"""兼容原始单文件名称的游戏入口。"""

from character import Character
from game import Game

__all__ = ["Character", "Game"]


if __name__ == "__main__":
    Game().start()
