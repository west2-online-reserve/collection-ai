n = int(input())

if n < 2 :
    print(False)

for x in range(2,n):
    if n % x == 0:
        print(False)
else:
    print(True)