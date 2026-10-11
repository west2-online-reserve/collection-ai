a=list(map(int,input().split()))
h=int(input())
s=int(h+30)
count=0

for i in a:
    if i<=s:
       count+=1

print(count)







