N=int(input())
W=True#当W变为False时才输出
for i in range(2,int(N**0.5)+1):
    if N%i==0:
        W=False
        break
if W:
    print("YES")
else:
    print("NO")