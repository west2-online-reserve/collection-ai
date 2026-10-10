from math import sqrt


def is_prime(num: int):
    for i in range(2, int(sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


num = int(input())
if is_prime(num):
    print("YES")
else:
    print("NO")
