n = int(input())
names=[""]
for i in range(n):
    name=input()
    names.append(name)
m = int(input())
for i in range(m):
    u, v=map(int, input().split())
    names[u] = "I_love_" + names[v]
print(names[1])
