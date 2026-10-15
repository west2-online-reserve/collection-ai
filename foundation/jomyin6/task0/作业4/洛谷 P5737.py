def isrun(year: int):
    if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
        return True
    else:
        return False


start_year, end_year = map(int, input().split(" "))
count = 0
leap_year = []
for year in range(start_year, end_year + 1):
    if isrun(year):
        count += 1
        leap_year.append(year)
print(count)
print(" ".join(map(str, leap_year)))
