apple = list(map(int,input().split()))
hight =int(input())
total=hight+30
count=0
for i in apple:
    if i<=total:
        count+=1
print(count)