num = list(map(int,input().split()))
high = int(input())
a = 0 
for i in num :
    if high+30 >= i :
        a+=1
print(a)