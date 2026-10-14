import random
# 1. 生成一副牌（共54张，不含花色）
ranks = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
deck = []
for rank in ranks:
    deck.extend([rank] * 4)
deck.append('小王')
deck.append('大王')

# 2. 洗牌
random.shuffle(deck)  

# 3. 发牌
player1 = deck[:17]       
player2 = deck[17:34]    
player3 = deck[34:51]     
others = deck[51:]        

# 4. 将结果写入文件
with open('player1.txt', 'w', encoding='utf-8') as f:
    f.write('玩家1的牌：\n')
    f.write(' '.join(player1)) 

with open('player2.txt', 'w', encoding='utf-8') as f:
    f.write('玩家2的牌：\n')
    f.write(' '.join(player2))

with open('player3.txt', 'w', encoding='utf-8') as f:
    f.write('玩家3的牌：\n')
    f.write(' '.join(player3))

with open('others.txt', 'w', encoding='utf-8') as f:
    f.write('底牌（多的三张）：\n')
    f.write(' '.join(others))

# 5. 在终端提示运行成功
print("发牌完成！请查看目录下生成的 player1.txt、player2.txt、player3.txt 和 others.txt 文件。")