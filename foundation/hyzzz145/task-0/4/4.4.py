N = int(input())
is_prime = 1
for i in range(2,N):
    if N % i == 0:
        is_prime = 0
        break
if is_prime:
    print("YES")
else:
    print("NO")
