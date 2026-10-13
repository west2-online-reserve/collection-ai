n = int(input())
names = [""]
for i in range(n):
    s = input()
    names.append(s)
m = int(input())
for _ in range(m):
    u, v = map(int, input().split())
    names[u] = "I_love_" + names[v]
print(names[1])
