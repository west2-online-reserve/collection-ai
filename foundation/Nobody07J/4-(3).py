x,y=map(int,input().split())
s=0
l=list()
for i in range(x,y+1):
    if (i%4==0 and i%100!=0) or i%400==0:
        l.append(i)
        s=s+1
print(s)
for item in l:
    print(item,end=" ")