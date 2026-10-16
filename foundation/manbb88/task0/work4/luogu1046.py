high = list(map(int, input().split()))
tall = int(input())
cnt = 0

for i in high:
    if tall + 30 >= i:
        cnt += 1

print(cnt)