import math

def is_prime(n):
    if n < 2:
        return False
    r = int(math.sqrt(n))
    for i in range(2, r+1):
        if n % i == 0:
            return False
    return True

N = int(input())
if is_prime(N):
    print("YES")
else:
    print("NO")
