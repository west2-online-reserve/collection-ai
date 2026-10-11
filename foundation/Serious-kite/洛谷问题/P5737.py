period = input("输入两个年份，空格分开:").split()  

a = int(period[0])
b = int(period[1])

count = 0

res = []

for year in range(a, b + 1):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        count += 1
        res.append(year)

print(count)
print(res)
