a=input().split()
b=int(a[0])
c=int(a[1])
d=0
e=[]
for i in range(b,c+1):
    if i % 400 == 0:
        d+=1
        e.append(i)
        continue
    if i % 4 == 0:
        if i % 100 != 0:
            d+=1
            e.append(i)
            continue        
print(d)
#for i in e:
#print(e)
print(*e)