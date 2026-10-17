s=int(input("请输入一个整数:"))
if s<2:
    print("NO")
elif s==2:
    print("YES")
else:
    for i in range(2,int(s**0.5)+1):
        if s%i==0:
            print("NO")
            break
        else:
            print("YES")
            break

