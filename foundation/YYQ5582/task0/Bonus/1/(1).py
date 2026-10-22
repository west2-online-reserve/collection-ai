x=int(input())
y=int(input())
z=int(input())
if x<y:
    if y<z:
        print(z,y,x)
    if y>z:
        if x>z:
            print(y,x,z)
        if x<z:
            print(y,z,x)
else:
    if x<z:
        print(z,x,y)
    if x>z:
        if z>y:
            print(x,z,y)
        if z<y:
            print(x,y,z)