a = "YES"
N = int(input())
for i in range(
    2, int(N**0.5) + 1
):  # 第一次在IDE做的时候没有+1导致误差，同时在豆包学习了开根号的做法
    if N % i == 0:
        a = "NO"
        break
print(a)
