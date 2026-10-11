from random import randint
import os


# -----------------卡牌数据-----------------
# 黑红方草1234
cards = [
[2, 1], [2, 2], [2, 3], [2, 4], 
[3, 1], [3, 2], [3, 3], [3, 4], 
[4, 1], [4, 2], [4, 3], [4, 4], 
[5, 1], [5, 2], [5, 3], [5, 4], 
[6, 1], [6, 2], [6, 3], [6, 4], 
[7, 1], [7, 2], [7, 3], [7, 4], 
[8, 1], [8, 2], [8, 3], [8, 4], 
[9, 1], [9, 2], [9, 3], [9, 4], 
[10, 1], [10, 2], [10, 3], [10, 4], 
['J', 1], ['J', 2], ['J', 3], ['J', 4], 
['Q', 1], ['Q', 2], ['Q', 3], ['Q', 4], 
['K', 1], ['K', 2], ['K', 3], ['K', 4], 
['A', 1], ['A', 2], ['A', 3], ['A', 4], 
['red_joker', 0], ['black_joker', 0]]

# 数字权重
weight_num = {2:1, 3:2, 4:3, 5:4, 6:5, 
               7:6, 8:7, 9:8, 10:9, 'J':10, 
               'Q':11, 'K':12, 'A':13, 
               'red_joker':13 ,'black_joker':14}
# 花色权重（黑>红>方>草）
weight_suit = {1:4, 2:3, 3:2, 4:1, 0:5}

player = [[],[],[]]
other_ = []


# 抽取
def draw(player_num):
    global player
    a = randint(0,53)
    while cards[a] == '':
        a = randint(0,53)
    player[player_num].append(cards[a])
    cards[a] = ''
    return


# 收纳未分配的卡牌
def other():
    global other_
    for x in range(0,54):
        if cards[x] != '':
            other_.append(cards[x])


# 发牌
for x in range(1,52):
    for y in range(0,3):
        if x % 3 == y:
            draw(y)
other()

# 排序
for t in range(0,3):
    print(t)
    player[t].sort(key=lambda x: (weight_num[x[0]],weight_suit[x[1]]), reverse = True)
other_.sort(key=lambda x: (weight_num[x[0]],weight_suit[x[1]]), reverse = True)

#输出发牌结果（终端）
print("p1:", player[0])
print("p2:", player[1])
print("p3:", player[2])
print("others:",other_)


# -----------------输出发牌结果（文件夹）-----------------
#获取文件所在位置
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
#创建文件夹
OUTPUT_FOLDER = os.path.join(BASE_PATH, "Bonus-8 The Cards")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

#获取发牌路径
p1_path = os.path.join(OUTPUT_FOLDER, "player1.txt")
p2_path = os.path.join(OUTPUT_FOLDER, "player2.txt")
p3_path = os.path.join(OUTPUT_FOLDER, "player3.txt")
others_path = os.path.join(OUTPUT_FOLDER, "others.txt")

# 调用文件
with open(p1_path, 'w', encoding="utf-8") as p1, \
     open(p2_path, 'w', encoding="utf-8") as p2, \
     open(p3_path, 'w', encoding="utf-8") as p3, \
     open(others_path, 'w', encoding="utf-8") as o:
    # 输出
    p1.write(str(player[0]))
    p2.write(str(player[1]))
    p3.write(str(player[2]))
    o.write(str(other_))

print("文件输出完成，文件夹：", OUTPUT_FOLDER)