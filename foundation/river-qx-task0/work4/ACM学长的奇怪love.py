name = []
n = int(input())
for i in range(0, n):
    name.append(input())
m = int(input())
for i in range(0, m):
    a, b = map(int, input().split())
    name[a - 1] = "I_love_" + name[b - 1]

print(name[0])
