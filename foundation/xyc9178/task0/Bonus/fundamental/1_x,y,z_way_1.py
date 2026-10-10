s = input().split()
x = int(s[0])
y = int(s[1])
z = int(s[2])
if x >= y:
    if x >= z:
        if y >= z:
            print(x,y,z)
        else:
            print(x,z,y)
    else:
        print(z,x,y)
elif y>=z:
    if x >= z:
        print(y,x,z)
    else:
        print(y,z,x)
else:
    print(z,y,x)