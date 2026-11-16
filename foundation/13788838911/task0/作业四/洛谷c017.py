import math
n=int(input())
flag=1
a=int(math.sqrt(n))
for i in range(2,a+1):
    if(n%i==0):
        flag=0
        break
if(flag==1):
    print("YES")
else:
    print("NO")