apples = list(map(int,input().split()))
h = int(input())
fh = h + 30
cnt = 0
for h1 in apples:
  if h1<=fh:
   cnt = cnt + 1
print(cnt)