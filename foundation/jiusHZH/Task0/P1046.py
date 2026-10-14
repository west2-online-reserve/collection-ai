a=list(map(int,input().split()))
b=int(input()) + 30
count=0
for i in a:
    if i<=b:
        count+=1
print(count)