a=eval(input())
b=[]
for i in a:
    if type(i)==int:
        b.append(i)
b.sort()
print(b)