N = int(input())
is_prime = True
if N <= 1:
    is_prime = False
else:
    for i in range(2, N):
        if N % i == 0:
            is_prime = False
            break

if is_prime == True:
    print("YES")
else:
    print("NO")
