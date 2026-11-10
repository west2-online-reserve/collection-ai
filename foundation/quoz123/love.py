n = int(input())
names = []
names.append("")          

for i in range(n):
    name = input()
    names.append(name)
m = int(input())

for i in range(m):
    u, v = input().split()
    u = int(u)
    v = int(v)
    names[u] = "I_love_" + names[v]
print(names[1])