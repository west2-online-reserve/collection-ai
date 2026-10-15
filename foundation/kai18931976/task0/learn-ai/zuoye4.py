n = int(input())

if n < 2:
    print("NO")
else:
    flag = 1  
    for i in range(2, n):
        if n % i == 0:
            flag = 0  
            break
    if flag == 1:
        print("YES")
    else:
        print("NO")

