x, y = map(int, input().split())
leap_years = [year for year in range(x, y + 1) if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]
print(len(leap_years))
print(' '.join(map(str, leap_years)))