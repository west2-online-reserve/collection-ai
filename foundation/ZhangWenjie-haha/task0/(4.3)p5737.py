a, b = map(int, input().split())

ans = []
cnt = 0
for i in range(a, b + 1):
    if (i % 4 == 0 and i % 100 > 0) or (i % 400 == 0):
        cnt += 1
        ans.append(i)

print(cnt)
for i in ans:
    print(i, end=' ')
