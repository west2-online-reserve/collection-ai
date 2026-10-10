N=int(input())
if 17<=N<=100000:
    for i in range(2,int(N**0.5)+1):
        if N%i!=0:
            print("YES")
            break
        else:
            print("NO")
            break