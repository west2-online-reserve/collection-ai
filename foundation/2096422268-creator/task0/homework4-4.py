import math
n=int(input())
def isprime(n):
    if n<2 or n%2==0:
        return False
    if n==2:
        return True
    for i in range(3,int(math.isqrt(n))+1,2):
        if n%i==0:
           return False
    return True
if isprime(n):
    print("YES")
else:
    print("NO")