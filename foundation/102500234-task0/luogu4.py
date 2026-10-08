num = int(input())
count = 0 
for i in range(1,num):
    if num%i == 0:
        count = count + 1
if count == 1:
    print("YES")
else:
    print("NO")