def comprise1(x, y, z):
    if x >= y >= z:
        print(x, y, z,end = " ")
    elif x >= z >= y:
        print(x, z, y,end = " ")
    elif y >= z >= x:
        print(y, z, x,end = " ")
    elif y >= x >= z:
        print(y, x, z,end = " ")
    elif z >= x >= y:
        print(z, x, y,end = " ")
    else:
        print(z, y, x,end = " ")

def comprise2(x, y, z):
    if x <= y:
        x, y = y, x
    if z <= y:
        y, z = z, y
    if z <= x:
        z, x = x, z
    print(x, y, z,end = ' ')

def comprise3(x, y, z):
    arr = [x, y, z]
    for i in range(len(arr)):
        if arr[i] <= arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
    for num in arr:
        print(num, end = ' ')

