num = int(input())
name = []
for i in range(0,num):
    name.append(input())
m = int(input())
for i in range(0,m):
    u_v = input().split()
    u = int(u_v[0])-1
    v = int(u_v[1])-1
    name[u] = "I_love_"+name[v]
print(name[0])