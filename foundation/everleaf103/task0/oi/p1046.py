s = input()
t = int(input())
h = list(map(int, s.split()))
ans = 0
for i in h:
    if 30 + t >= i:
        ans += 1
print(ans)
