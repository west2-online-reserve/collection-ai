b = input().split()
c = int(input())
c += 30
d, e = 0, 0
while d < 10:
    if int(b[d]) <= c:
        e += 1
    d += 1
print(e)
