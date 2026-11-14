import math
# math.sqrt(x): 返回浮点数
# math.isqrt(x): 返回整数, 向下取整 = int(math.sqrt(x))
def isPrime(n):
    if n % 2 == 0:
        return False
    for i in range(3, math.isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True

num = int(input())
print('YES' if isPrime(num) else 'NO')