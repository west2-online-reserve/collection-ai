def leap(year):
    return(year%4==0 and year%100!=0) or (year%400==0)
x,y=map(int,input().split())
ly=[year for year in range(x,y+1) if leap(year)]
print(len(ly))
print(" ".join(map(str,ly)))