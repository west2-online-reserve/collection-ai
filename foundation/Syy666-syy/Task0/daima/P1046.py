apple=list(map(int,input().split()))
tao=int(input())
total=tao+30
count=0
for h in apple:
    if h<=total:
        count+=1
print(count)