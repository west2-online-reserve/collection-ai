apple=list(map(int,input().split()))
height=int(input())
height+=30
count=0
for i in apple:
    if i<=height:
        count+=1
print(count)