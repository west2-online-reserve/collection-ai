n = int(input())
names = [""] * n
for i in range(n):
    names[i] = input()
m = int(input())
for t in range(m):
    u, v = map(int, input().split())
    names[u - 1] = "I_love_" + names[v - 1]
print(names[0])