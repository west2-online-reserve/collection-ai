import math
N = int(input())
check = 1
for i in range  (2,int(math.sqrt(N))+1):
    if N % i == 0:
        check = 0
if check ==1 :
    print("YES")
else:
    print("NO")