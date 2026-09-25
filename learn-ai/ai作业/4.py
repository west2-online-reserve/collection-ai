x,y=map(int,input().split())
count=0
m=[]
for i in range(x,y+1):
    if i%4==0 and i%100!=0 or i%400==0:
        count+=1
        m.append(i)
print(count)
print(*m)