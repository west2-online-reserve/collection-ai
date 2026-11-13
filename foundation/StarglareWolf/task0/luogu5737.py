que = list(map(int, input().split()))
y = que.pop()
x = que.pop()
total = 0
years = []
for year in range(x , y + 1):
    if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
        years.append(year)
        total += 1
print(total)
for year in years:
    print(year,end = " ")