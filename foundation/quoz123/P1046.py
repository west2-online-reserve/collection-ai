heights = list(map(int, input().split()))
reach = int(input())
limit = reach + 30
count = 0
for h in heights:
    if h <= limit:
        count += 1
print(count)