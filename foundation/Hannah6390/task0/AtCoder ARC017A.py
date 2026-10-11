N=int(input())
prime = True

for i in range(2,N):
    if N%i==0:
        prime = False
        break

if prime:
     print("YES")
else:
     print("NO")