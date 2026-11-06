s=int(input())
true=0
for i in range(2,int(s**0.5)+1):
    if s%i==0:
        true=1
        break
if true:
    print("NO")
else:
    print("YES")