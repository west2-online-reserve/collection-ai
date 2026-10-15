apples = list(map(int, input().split()))
h = int(input())
max_h = h + 30
count = 0
for a in apples:
    if a <= max_h:
        count += 1
print(count)

