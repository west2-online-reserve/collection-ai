import re
a=input()
a=re.sub(r"\D+","  ",a)
a=a.split()
print(a)
b=[]
for i in a:
    b.append(int(i))
print(b)    
for i in range(len(b)-1):
    for j in range(len(b)-1):
        if b[j] > b[j+1]:
            b[j],b[j+1]=b[j+1],b[j]
print(b)            
