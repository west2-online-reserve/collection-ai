a=input ()
num=a.split()
b=input()
name=b.split()
d=dict(zip (num,name))
for i in num:
    if int (i)%2==0:
        del d[i]
print(d)
