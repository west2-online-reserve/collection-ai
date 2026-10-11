year=input()
a,b=map(int,year.split())
years=list(range(a,b+1))
num=0
j=[]
for i in years:
    if (i%4==0 and i%100!=0) or (i%400==0):
        num=num+1
        j.append(i)
print(num)
print(*j)
