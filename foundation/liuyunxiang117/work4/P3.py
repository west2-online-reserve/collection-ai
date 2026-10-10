x,y = list(map(int, input().split()))
count=0
for i in range(x,y+1):
    if i%4==0  and i%100!=0 or i%400==0:
        print(i,end=" ")
        count+=1
print()
print(count)