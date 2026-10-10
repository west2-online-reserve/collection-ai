apples = input().split()
h = int(input())
count = 0
for i in range(10):
    if h + 30 >= int(apples[i]):
        count += 1
print(count)