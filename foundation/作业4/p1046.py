h = int(input())
apples = list(map(int, input().split()))
count = 0
for height in apples:
    if height <= h + 30:
        count += 1
print(count)