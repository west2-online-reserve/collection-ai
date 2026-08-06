import math
a=int(input())
if a%2==0:
    print("NO")
    exit()
max1=int(math.sqrt(a))+1
for x in range(3,max1,2):
    if a%x==0:
        print("NO")
        exit()
print("YES")