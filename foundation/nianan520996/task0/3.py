height=list(map(int,input().split()))
tao=int(input())
tao_max=tao+30
count=0
for i in height:
    if i <= tao_max:
        count += 1
print(count)