years: list[str] = input().split()
start_year: int = int(years[0])
end_year: int = int(years[1])
count = 0

leap_years: list[int] = []

for year in range(start_year, end_year + 1):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                count += 1
                leap_years.append(year)
        else:
            count += 1
            leap_years.append(year)

result = " ".join(map(str, leap_years))

print(count)

if count > 0:
    print(result)
