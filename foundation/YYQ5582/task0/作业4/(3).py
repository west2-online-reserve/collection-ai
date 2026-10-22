n=input().split()
t=[]
a=0
for i in range(int(n[0]),int(n[1])+1):
    if (i%4==0 and i%100!=0) or i%400==0:
        a+=1
        t.append(i)
    else:
        continue
print(a)
for num in t:
    print(num,end=" ")



