import math

n = int(input())

if n < 2:
    print("NO")
elif n == 2:
    print("YES")
elif n % 2 == 0:
    print("NO")
else:
    is_prime = True
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            is_prime = False
            break
    print("YES" if is_prime else "NO")