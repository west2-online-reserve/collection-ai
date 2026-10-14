import sys


def main():
    data = sys.stdin.read().split()  # 把整个输入按空白切成一串词
    idx = 0  # idx 是"下一个还没读的词的编号"

    n = int(data[idx])
    idx += 1  # 手指指到 '5'，读完往后挪 → n=5

    name = [""] * (n + 1)  # 开 6 个格子: name[0]~name[5]
    for i in range(1, n + 1):
        name[i] = data[idx]
        idx += 1  # name[1]='anonymous', name[2]='natalia',

    m = int(data[idx])
    idx += 1
    for _ in range(m):  # 循环 m 次，每轮处理一条记录
        u = int(data[idx])
        v = int(data[idx + 1])
        idx += 2
        # 用 v 当前的名字更新 u
        name[u] = "I_love_" + name[v]

    print(name[1])


if __name__ == "__main__":
    main()  # reference from deepseek
