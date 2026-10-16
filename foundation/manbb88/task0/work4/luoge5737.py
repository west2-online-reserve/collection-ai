x,y = map(int, input().split())
l = []
cnt = 0
for i in range(x, y + 1):
    if i % 4 == 0 and i % 100 != 0 or i % 400 == 0:
        l.append(i)
        cnt += 1
print(cnt)
for i in l:
    print(i, end=' ')