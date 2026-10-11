arr =list(map(int,input().split()))
n=int(input())
cnt=0
for x in arr:
    if x<=n+30:
        cnt=cnt+1
print(cnt)