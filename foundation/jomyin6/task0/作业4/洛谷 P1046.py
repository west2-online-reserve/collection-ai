a = list(map(int, input().split(" ")))
height = int(input())
count = 0
for h in a:
    if h <= height + 30:
        count += 1
print(count)
