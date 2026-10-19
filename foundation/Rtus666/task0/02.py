a=list(map(int,input().split()))
h=int(input())
th=h+30
c=sum(1 for  i in a if i<=th)
print(c)