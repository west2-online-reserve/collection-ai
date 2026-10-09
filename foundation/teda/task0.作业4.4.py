N = int(input())

is_prime = True
if N < 2:
    is_prime = False

for i in range(2, int(N ** 0.5) + 1):
    if N % i == 0:
        is_prime = False
        break   

if is_prime:
    print("YES")
else:
    print("NO")