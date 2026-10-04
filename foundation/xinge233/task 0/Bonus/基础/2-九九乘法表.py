#九九乘法表
i=1
j=1
while i<=9:
    print(j,"*",i,"=",i*j,end=" ")
    j+=1
    if j>i:
        print()
        i+=1
        j=1
        