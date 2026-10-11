import random

dic = {"spade": 0.1, "heart": 0.2, "club": 0.3, "diamond": 0.4}
color = ["spade", "heart", "club", "diamond"]
num = [x for x in range(1, 14)]
total = [(i, x) for i in color for x in num]
total += [("Joker1", 17), ("Joker2", 16)]
random.shuffle(total)
a, b, c = [total[i * 17 : (i + 1) * 17] for i in range(3)]
d = total[51:54]


def way(t):
    f = t[1]
    try:
        s = dic[t[0]]
    except KeyError:
        s = 0
    if f == 1:
        f = 14
    elif f == 2:
        f = 15
    return f + s


def paixu(t):
    t = sorted(t, key=way, reverse=True)
    return t


with open("player1.txt", "w") as f:
    f.write(str(paixu(a)) + "\n")
with open("player2.txt", "w") as f:
    f.write(str(paixu(b)) + "\n")
with open("player3.txt", "w") as f:
    f.write(str(paixu(c)) + "\n")
with open("others.txt", "w") as f:
    f.write(str(paixu(d)) + "\n")
# 题目要求放在这几个文件里 我只是做题 所以就没建对应的文件