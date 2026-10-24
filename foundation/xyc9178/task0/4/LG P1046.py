s = input().split()
b = int(input())
sum = 0
for i in s:
    i=int(i)
    if i <= (b+30):
        sum += 1
print(sum)