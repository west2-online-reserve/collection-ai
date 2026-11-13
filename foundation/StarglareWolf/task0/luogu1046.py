nums = list(map(int, input().split()))
achieve = int(input())
achieve += 30
total = 0
for num in nums:
    if achieve >= num:
        total += 1
print(total)