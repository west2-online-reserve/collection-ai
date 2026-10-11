n = int(input())
name = [''] * (n + 1)
num = [0] * (n + 1)

for i in range(1, n + 1):
    name[i] = input()
m = int(input())

for _ in range(m):
    a, b = map(int, input().split())
    name[a] = name[b]
    num[a] = num[b] + 1

print("I_love_" * num[1] + name[1])
