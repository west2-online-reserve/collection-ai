N=int(input())
if N in range(17,1000001):
    for i in range(17,N+1):
        if N!=17 and N%i==0:
            print('NO')
            break
        if N==17 or N%i!=0:
            print('YES')
            break
