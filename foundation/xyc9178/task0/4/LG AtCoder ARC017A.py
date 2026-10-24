import math
N = int(input())
check = True
for i in range  (2,int(math.sqrt(N))+1):
    if N % i == 0:
        check = False
if check :
    print("YES")
else:
    print("NO")