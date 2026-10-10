n=int(input())
name=[""]
for i in range(n):
    name.append(input())
m=int(input())
for j in range(m):
    u,v=map(int,input().split())
    name[u]="I_love_"+name[v]
print(name[1])