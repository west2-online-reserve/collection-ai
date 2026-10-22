h=input().split()
t=int(input())
l=t+30
n=0
for i in h:
    H=int(i)
    if H<=l:
        n+=1
    else:
        continue
print(n)

