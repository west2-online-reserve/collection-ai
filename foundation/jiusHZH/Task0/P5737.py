a=list(map(int,input().split()))
def check(year):
    if year%4 ==0 and year%100 !=0 or year%400 ==0:
        return True
    return False
b=a[1]
a=a[0]
year=[]
for i in range(a,b+1):
    if check(i):
      year.append(i)
print(len(year))
print(" ".join(map(str,year)))