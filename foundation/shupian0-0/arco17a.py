n = int(input())
ok = True
for i in range(2, n):
    if n % i == 0:
        ok = False
        break
if ok:
    print("YES")
else:
    print("NO")