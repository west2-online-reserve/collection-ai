l=[]
n=input().split()
for i in n:
    try:
        a=int(i)
        l.append(a)
    except:
        pass
l.sort(reverse=False)
print(l)