n = int(input())

flag = True

if n < 2:
    flag = False
else:
    i = 2
    while i < n:
        if n % i == 0:
            flag = False
            break
        i = i + 1

if flag == True:
    print("YES")
else:
    print("NO")