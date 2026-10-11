n = int(input())
x = int(n**0.5)
judge = 1
for i in range(2, x + 1):
    if n % i == 0:
        judge = 0
        break
if judge:
    print("YES")
else:
    print("NO")
