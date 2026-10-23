x, y, z = map(int, input().split())

tempx = x
if x < y :
    x = y
    y = tempx
    tempx = x
if x < z :
    x = z
    z = tempx
    tempx = x

tempy = y
if y < z :
    y = z
    z = tempy

print(x, y, z)
