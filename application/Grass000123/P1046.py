a=list(map(int, input().split()))
h = int(input())
maxh = h + 30
count = 0
for num in a:
    if num <= maxh:
        count +=1
print(count)
