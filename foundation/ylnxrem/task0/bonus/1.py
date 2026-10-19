x, y, z = map(int, input().split())
if x < y:
    x, y = y, x
if x < z:
    x, z = z, x
if y < z:
    y, z = z, y
print(x, y, z)

# or
a = list(map(int, input().split()))
a.sort(reverse=True)
print(a)
