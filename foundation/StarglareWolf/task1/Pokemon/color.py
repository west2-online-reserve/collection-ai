RESET   = "\033[0m"
OPPONENT     = "\033[91m"
PLAYER    = "\033[94m"

POSITION = {
    "player":"玩家",
    "opponent":"电脑"
}

def color(text: str, color: str) -> str:
    return f"{color}{text}\033[0m"
