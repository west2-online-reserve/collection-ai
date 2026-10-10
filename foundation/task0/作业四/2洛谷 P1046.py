tree = list(map(int,input().split()))
h = int(input())
cnt = 0
for i in tree:
    if i <= (h+30):
        cnt=cnt+1
print(cnt)