s = input().split()
a = int(s[0])
b = int(s[1])
arr = []
num = 0
for i in range(a,b+1):
    if (i%4 ==0 and i%100 != 0) or i%400 == 0:
        arr.append(i)
        num = num+1
print(num)
for i in range(len(arr)):
    print(arr[i],end=" ")
        








