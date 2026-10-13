x, y = map(int,input().split())
leaplist = []
for year in range(x,y+1):
    if(year%4==0 and year % 100 != 0) or (year%400==0):
        leaplist.append(year)
print(len(leaplist))
print(' '.join(map(str, leaplist)))