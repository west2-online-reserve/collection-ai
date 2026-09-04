apples=list(map(int,input().split()))
#获取整数列表
height=int(input())
height+=30
count=0
for apple in apples:
    if apple<=height:
        count+=1

print(count)