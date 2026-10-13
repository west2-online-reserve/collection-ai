s = input()
x, y = s.split()
x = int(x)
y = int(y)
years = []
count = 0
for year in range(x,y+1):
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        count = count +1
        years.append(year)
print(count)
print(*years)