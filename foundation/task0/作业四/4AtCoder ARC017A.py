import math
n = float(input())
if n==1:
    print("NO")
    exit()
for i in range(2,int(math.sqrt(n))+1):
    if(n%i==0):
        print("NO")
        exit()
print("YES")