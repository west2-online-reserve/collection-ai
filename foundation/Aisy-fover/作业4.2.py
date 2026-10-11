s=input()
a=list(map(int,s.split()))
b=int(input())+30
j=0
for i in a:
    if i<=b:
        j=j+1
print(j)
