a=list(map(int,input().split()))
b=int(input())
num=0
for x in a:
    if x<=b+30:
      num=num+1
print(num)