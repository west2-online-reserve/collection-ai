n = int(input())
name = [""]*(n+1)
for i in range(1,n + 1):
    name[i] = input()

m = int(input())
for i in range(m):
    u, v = map(int,input().split())
    name[u] = "I_love_" + name[v]

print(name[1])