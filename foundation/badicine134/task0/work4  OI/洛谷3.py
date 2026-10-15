year_list = []
num = 0
x, y = [int(i) for i in input().split()]
for i in range(x, y+1):
    if (i % 4 == 0 and i % 100 != 0) or (i % 400) == 0:
        num += 1
        year_list.append(i)
print(num)
print(*year_list)