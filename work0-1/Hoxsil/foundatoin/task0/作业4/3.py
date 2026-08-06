def lunnian(year):
    if year%100==0:
        if year%400==0:
            return 1
        else:
            return 0
    else:
        if year%4==0:
            return 1
        else:
            return 0
a,b=map(int,input().split())
num=0
list1=[]
for x in range(a,b+1):
    if lunnian(x):
        num=num+1
        list1.append(x)
print(num)
print(*list1)