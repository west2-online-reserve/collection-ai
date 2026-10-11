n=int(input())
names=[""]*(n+1)
for i in range(1,n+1):
    names[i]=input().strip()
m=int(input())
for j in range(m):
    u,v=map(int,input().split())
    names[u]="I_love_"+names[v]
print(names[1])
