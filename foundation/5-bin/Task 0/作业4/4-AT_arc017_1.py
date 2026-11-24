import sys

n = int(input())
for i in range(2, n):
    if n % i == 0:
        print("NO")
        sys.exit(0)
print("YES")