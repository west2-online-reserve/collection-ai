arr = list(map(int, input().split()))
h0 = int(input()) + 30

cnt = 0
for x in arr:
    if h0 >= x:
        cnt += 1

print(cnt)