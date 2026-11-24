from pathlib import Path
from random import shuffle


def main():
    cards = []

    for i in range(2, 11):
        for j in range(0, 4):
            cards.append(str(i))

    for i in ["J", "Q", "K", "A"]:
        for j in range(0, 4):
            cards.append(i)

    # append()一次只能压入一个元素
    cards.append("大王")
    cards.append("小王")
    # extend(["大王", "小王"])一次可以加入多个元素
    # 将列表中元素拆成单个元素，分别加入
    # cards.extend(["大王", "小王"])

    shuffle(cards)

    # 获取当前脚本所在目录，使输出路径不受终端当前目录影响
    output_dir = Path(__file__).parent

    with open(output_dir / "player1.txt", "w", encoding="UTF-8") as file:
        file.write(" ".join(cards[0:17]))
    with open(output_dir / "player2.txt", "w", encoding="UTF-8") as file:
        file.write(" ".join(cards[17:34]))
    with open(output_dir / "player3.txt", "w", encoding="UTF-8") as file:
        file.write(" ".join(cards[34:51]))
    with open(output_dir / "others.txt", "w", encoding="UTF-8") as file:
        file.write(" ".join(cards[51:54]))


if __name__ == "__main__":
    main()
