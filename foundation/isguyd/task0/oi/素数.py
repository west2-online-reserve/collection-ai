a=int(input())
b=int(a**0.5)
for i in range(2,b):
    if a%i==0:
        print("NO")
        break
        sys.exit(0)
print("YES")