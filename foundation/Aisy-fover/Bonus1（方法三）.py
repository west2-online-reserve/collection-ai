x=eval(input("请输入一个数："))
y=eval(input("请输入一个数："))
z=eval(input("请输入一个数："))
a=x,y,z
b=sorted(a,reverse=True)
print(*b)
