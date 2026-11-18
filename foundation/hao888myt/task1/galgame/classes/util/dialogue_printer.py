import sys
import time
from typing import Any

try:
    from msvcrt import getch, kbhit  # Windows
except ImportError:
    import select
    import termios
    import tty

    def kbhit():
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(dr)

    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)  # type: ignore
        try:
            tty.setraw(fd)  # type: ignore
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)  # type: ignore


PAUSE_CHARS = {
    "，": 6.0,
    "。": 8.0,
    "！": 8.0,
    "？": 8.0,
    "…": 4.0,
    "、": 4.0,
    "；": 8.0,
    "：": 8.0,
}


class DialoguePrinter:
    def __init__(self, delay: float = 0.05):
        self.delay = delay

    def print_line(self, line: dict[str, Any]):
        """打印一句对话（带角色名）"""
        name = line.get("name", "")
        content = line.get("content", "")

        if name:
            sys.stdout.write(f"[{name}] ")
            sys.stdout.flush()

        self.typewriter(content)
        print()

    def typewriter(self, text: str):
        """逐字打印，标点减速，按键跳过"""
        for i, char in enumerate(text):
            if kbhit():
                getch()
                sys.stdout.write(text[i:])
                sys.stdout.flush()
                break

            sys.stdout.write(char)
            sys.stdout.flush()

            pause = PAUSE_CHARS.get(char, 1.0)
            time.sleep(self.delay * pause)

    def print_choices(self, choices: list[dict[str, Any]]):
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c['content']}")
