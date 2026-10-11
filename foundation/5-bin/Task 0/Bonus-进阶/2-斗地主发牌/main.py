from pathlib import Path
from random import shuffle


def main():
    cards = []

    for i in range(3, 11):
        for j in range(0, 4):
            cards.append((str(i), i))

    k = 11
    for i in ["J", "Q", "K", "A", "2"]:
        for j in range(0, 4):
            cards.append((i, k))
        k += 1

    # append()一次只能压入一个元素
    cards.append(("大王", 17))
    cards.append(("小王", 16))
    # extend(["大王", "小王"])一次可以加入多个元素
    # 将列表中元素拆成单个元素，分别加入
    # cards.extend(["大王", "小王"])

    shuffle(cards)

    # 获取当前脚本所在目录，使输出路径不受终端当前目录影响
    output_dir = Path(__file__).parent

    with open(output_dir / "player1.txt", "w", encoding="UTF-8") as file:
        player1_cards = sorted(cards[0:17], key=lambda x: x[1], reverse=True)
        file.write(" ".join(card[0] for card in player1_cards))
    with open(output_dir / "player2.txt", "w", encoding="UTF-8") as file:
        player2_cards = sorted(cards[17:34], key=lambda x: x[1], reverse=True)
        file.write(" ".join(card[0] for card in player2_cards))
    with open(output_dir / "player3.txt", "w", encoding="UTF-8") as file:
        player3_cards = sorted(cards[34:51], key=lambda x: x[1], reverse=True)
        file.write(" ".join(card[0] for card in player3_cards))
    with open(output_dir / "others.txt", "w", encoding="UTF-8") as file:
        others_cards = sorted(cards[51:54], key=lambda x: x[1], reverse=True)
        file.write(" ".join(card[0] for card in others_cards))


if __name__ == "__main__":
    main()
