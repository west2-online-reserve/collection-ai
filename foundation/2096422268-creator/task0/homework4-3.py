x,y=map(int,input().split())
leap_years=[]
for year in range(x,y+1):
    if (year%4==0 and year%100!=0) or (year%400==0):
        leap_years.append(year)
print(len(leap_years))
print(*leap_years)