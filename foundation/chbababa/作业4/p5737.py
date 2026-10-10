x, y = map(int, input().split())
ans = []
for year in range(x, y + 1):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        ans.append(str(year))
print(' '.join(ans))