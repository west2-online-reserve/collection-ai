a, b = map(int, input().split())
ans = 0
year = []
for i in range(a, b + 1):
    if i % 4 == 0 and i % 100 != 0 or i % 400 == 0:
        ans += 1
        year.append(i)
print(ans)
for i in year:
    print(i, end=" ")
