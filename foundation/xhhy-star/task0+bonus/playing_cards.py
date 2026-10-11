import random

def main():
    base_cards = [ '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']

    deck = []

    for card in base_cards:
        for i in range(4):
            deck.append(card)

    deck.append('小王')
    deck.append('大王')

    
# 这里我验证过了，是使用print(len(deck))，请学长注意喵

    random.shuffle(deck)
    print("洗牌后的牌库:", deck)

    player1 = deck[:17]
    player2 = deck[17:34]
    player3 = deck[34:51]
    others = deck[51:54]

# 此处我也验证了喵，代码如下，print(len(player1), len(player2), len(player3), len(others))，请学长注意喵

    files = ['player1.txt', 'player2.txt', 'player3.txt', 'others.txt']  #这个询问了ai，用列表存储文件名，方便后续循环写入文件（可用open（）函数写入）
    players_data = [player1, player2, player3, others]

    for file, data in zip(files, players_data):
        with open(file, 'w', encoding='utf-8') as f:    #这个encoding='utf-8'是为了防止写入中文时出现乱码（我询问了ai，特此声明）
            f.write(', '.join(data))      #此处是将列表中的元素用逗号和空格连接成一个字符串，并写入文件中（新学的嘻嘻）

            print("文件录入完毕！")

if __name__ == "__main__":
    main()
