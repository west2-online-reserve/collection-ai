a = list(map(int, input().split()))
b = int(input())
cnt = 0
for num in a:
    if num <= b+30:
        cnt +=1
print(cnt)