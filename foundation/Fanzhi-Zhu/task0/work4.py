#洛谷P1001.py
data=input().split()
sum=int(data[0])+int(data[1])
print(sum)

#洛谷P1046.py
data=input().split()
height=int(input())
count=0
for i in range(10) :
    if int(data[i])<=height+30 :
        count+=1
print(count)

#洛谷 P5737.py
data=input().split()
x=int(data[0])
y=int(data[1])
count=0
result=[]
for i in range(x,y+1) :
      if (i%4==0 and i%100!=0) or (i%400==0) :
          count+=1
          result.append(i)
print(count)          
print(' '.join(map(str,result)))

#AtCoder ARC017A.py
def sushu(x):
    if x<=1 :
        return False
    for i in range(2,int(x**0.5)+1):
        if x%i==0:
            return False
    return True
n=int(input())
if sushu(n):
    print("YES")
else:
    print("NO")

#ACM集训队.py
n=int (input())
names=[None]*(n+1)

for i in range(1,n+1):
    names[i]=input()

m=int (input())

for _ in range(m):
    u,v=map(int,input().split())
    names[u]="I_love_"+names[v]

print(names[1])