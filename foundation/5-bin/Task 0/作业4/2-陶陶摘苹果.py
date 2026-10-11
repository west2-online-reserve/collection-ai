lis = list(map(int,input().split()))
h1 = int(input())
h2 = 30
sum = 0
for i in range(len(lis)):
    if h1 + h2 >= lis[i]:
        sum+=1
print(sum)