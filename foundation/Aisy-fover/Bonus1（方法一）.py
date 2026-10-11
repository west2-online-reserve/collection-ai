x=eval(input("请输入一个数："))
y=eval(input("请输入一个数："))
z=eval(input("请输入一个数："))
if x>=y:
    if y>=z:
        print(x,y,z)
    elif z>=x:
        print(z,x,y)
    else:
        print(x,z,y) 
elif y>x:
    if z>=y:
        print(z,y,x)
    elif y>=z:
        print(y,z,x)
    else:
        print(y,x,z)
