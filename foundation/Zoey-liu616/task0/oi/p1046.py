apples = list(map(int, input().split()))
height = int(input())
count = 0
for apple in apples:
    if apple <= height + 30:
        count += 1

print(count)