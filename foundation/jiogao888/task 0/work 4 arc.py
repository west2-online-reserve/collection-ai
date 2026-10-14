import math
num = int(input())
result = True
for i in range(2,math.floor(num**0.5)+1):
    if num%i==0:
        result = False
        break
if result:
    print('YES')
else:
    print("NO")