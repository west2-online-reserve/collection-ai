def prime(N):
    if N == 0 or N == 1:
        return "NO"
    x = N ** 0.5
    for i in range(2, int(x) + 1):
        if N % i == 0:
           return "NO"
    return "YES"

num = int(input())
print(prime(num))