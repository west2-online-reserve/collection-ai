import math

num=int(input())
if num%2==0:
    print('NO')
else:
    for i in range(3,int(math.sqrt(num))+1,2):
        if num%i==0:
            print('NO')
            break
    else:
        print('YES')