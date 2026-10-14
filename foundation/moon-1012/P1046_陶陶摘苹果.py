apples=list(map(int,input().split()))
h=int(input())
m=h+30
cnt=0
for i in apples:
    if i<=m:
        cnt+=1
print(cnt)
