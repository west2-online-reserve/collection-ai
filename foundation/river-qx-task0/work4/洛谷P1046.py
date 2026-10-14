apples = list(map(int, input().split()))

tao_high = int(input())

count = 0

for a in apples:
    if a <= tao_high + 30:
        count += 1

print(count)
