# 作业 4 - 第 5 题：I_love_ 学长名字
# 思路与 C++ 版完全一致：不去真正拼接字符串，只记录两件事
#   1) count[i]：编号 i 当前的名字里有多少个 "I_love_" 前缀
#   2) root[i] ：最里层用的是哪个人的初始名字
# 每次「u 喜欢上 v」：count[u] = count[v] + 1, root[u] = root[v]
# 复杂度 O(n + m)，不受名字长度膨胀的影响。

import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    n = int(data[pos])
    pos += 1

    name = [""] * (n + 1)
    for i in range(1, n + 1):
        name[i] = data[pos]
        pos += 1

    m = int(data[pos])
    pos += 1

    count = [0] * (n + 1)      # count[i]: 名字里 "I_love_" 的个数
    root = list(range(n + 1))  # root[i] : 最里层的初始名字属于谁

    for _ in range(m):
        u = int(data[pos])
        v = int(data[pos + 1])
        pos += 2
        count[u] = count[v] + 1
        root[u] = root[v]

    # 编号 1 的学长最后所拥有的名字
    sys.stdout.write("I_love_" * count[1] + name[root[1]] + "\n")


main()