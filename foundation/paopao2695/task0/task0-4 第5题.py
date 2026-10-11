n=int(input())
name=[]
for i in range(n):
    name.append(input())

m=int(input())
for i in range(m):
    p=input().split()
    u=int(p[0])
    v=int(p[1])
    name[u-1]="I_love_"+name[v-1]

print(name[0])