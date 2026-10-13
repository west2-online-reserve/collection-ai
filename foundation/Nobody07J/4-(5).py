dic=dict()
n=int(input())
for i in range(1,n+1):
    s=str(input())
    dic[i]=s
m=int(input())
for a in range(1,m+1):
    b,c=map(int,input().split())
    dic[b]="I_love_"+dic[c]
print(dic[1])