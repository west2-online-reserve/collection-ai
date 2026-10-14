n=int(input())
name=[]
for i in range(n):
    name.append(input())
count=0
m=int(input())
while m>0:
    u,v=map(int,input().split())
    name[u-1]='I_love_'+name[v-1]
    m-=1
print(name[0])