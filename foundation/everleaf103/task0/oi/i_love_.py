n = int(input())
name = []
for i in range(n):
    str = input()
    name.append(str)
m = int(input())
for i in range(m):
    u, v = map(int, input().split())
    name[u - 1] = "I_love_" + name[v - 1]
print(name[0])
