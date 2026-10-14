import math

N = int(input())
flag = True
if N < 2:
    flag = False
else:
    for i in range(2, int(math.sqrt(N)) + 1):
        if N % i == 0:
            flag = False
            break
if flag:
    print("YES")
else:
    print("NO")
