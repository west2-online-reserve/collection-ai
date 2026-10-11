n = int(input())
names = []
for i in range(n):
    names.append(input())

m = int(input())
for i in range(m):
    u_v = input().split()
    u = int(u_v[0])
    v = int(u_v[1])
    names[u - 1] = "I_love_" + names[v - 1]

print(names[0])