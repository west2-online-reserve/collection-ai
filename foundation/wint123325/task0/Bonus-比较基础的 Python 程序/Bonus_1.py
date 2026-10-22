x=int(input())
y=int(input())
z=int(input())
if y>x:
    x,y=y,x
if z>x:
    z,x=x,z
if z>y:
    z,y=y,z
print(x,y,z)