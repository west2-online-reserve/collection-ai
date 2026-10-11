x,y=map(int,input().split())
years=[]
for i in range(x,y+1):
    if (i%4==0 and i%100!=0) or i%400==0:
        years.append(i)
print(len(years))
print(*years)