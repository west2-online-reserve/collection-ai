import random

# 1. 定义牌面的大小顺序（从小到大）
ranks = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', '小王', '大王']

# 2. 建立一个权重字典：牌面 -> 数字大小
rank_weight = {rank: i for i, rank in enumerate(ranks)}

# 3. 生成一副牌（54张）
deck = []
for rank in ranks[:-2]:
    deck.extend([rank] * 4)
deck.append('小王')
deck.append('大王')

# 4. 洗牌
random.shuffle(deck)

# 5. 发牌
p1 = deck[:17]
p2 = deck[17:34]
p3 = deck[34:51]
others = deck[51:]

# 6. 自定义排序函数：根据权重从大到小排序
def sort_cards(cards):
    return sorted(cards, key=lambda x: rank_weight[x], reverse=True)

# 对每个人的牌进行排序
p1 = sort_cards(p1)
p2 = sort_cards(p2)
p3 = sort_cards(p3)
others = sort_cards(others)

# 7. 定义写文件的函数
def save_to_file(filename, cards):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(' '.join(cards))

# 8. 分别写入四个文件
save_to_file('player1.txt', p1)
save_to_file('player2.txt', p2)
save_to_file('player3.txt', p3)
save_to_file('others.txt', others)

print("发牌完成！请查看目录下生成的 player1.txt、player2.txt、player3.txt 和 others.txt 文件。")