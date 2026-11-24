from game import Game


def main() -> None:
    """创建游戏对象并启动游戏。"""
    game: Game = Game()
    game.start()


if __name__ == "__main__":
    main()
