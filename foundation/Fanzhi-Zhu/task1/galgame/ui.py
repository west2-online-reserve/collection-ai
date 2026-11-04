"""输入输出相关的辅助函数。

把 `input()` 的异常处理和「非法输入重问」集中在这里，
游戏逻辑里就不需要到处写 `while True` 了。
"""

from typing import Final

INVALID_HINT: Final[str] = "无效输入，请重新选择。"


def read_line(prompt: str) -> str:
    """读取一行输入；遇到 EOF 或 Ctrl+C 时优雅地结束游戏。"""
    try:
        return input(prompt)
    except EOFError:
        print("\n（输入已结束，游戏自动退出。）")
        raise SystemExit(0)
    except KeyboardInterrupt:
        print("\n（游戏已中断。）")
        raise SystemExit(0)


def ask_choice(prompt: str, valid: set[str]) -> str:
    """反复询问，直到玩家输入 `valid` 中的某个选项（大小写不敏感）。"""
    while True:
        answer = read_line(prompt).strip().upper()
        if answer in valid:
            return answer
        print(INVALID_HINT)


def divider(title: str = "") -> None:
    """打印一条分隔线，可选标题。"""
    line = "=" * 46
    if title:
        print(f"\n{line}\n【{title}】\n{line}")
    else:
        print(f"\n{line}")
