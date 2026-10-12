nums = list(map(int, input().split()))
nums.sort(reverse=True)
print(*nums)