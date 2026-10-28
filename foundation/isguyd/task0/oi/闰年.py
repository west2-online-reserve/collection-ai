a,b = map(int,input().split())
sum = 0
alist=[]
for i in range(b-a+1):
    if((a + i) % 4 == 0 and (a + i) % 100 !=0):
        alist.append(a+i)
        sum+=1
    elif((a + i)%400 == 0):
        alist.append(a+i)
        sum+=1
print(sum,end="\n")
for i in range(sum):
    print(alist[i],end=" ") 