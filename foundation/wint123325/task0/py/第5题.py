a=int(input())
b={}
for i in range(1,a+1):
    b[i]=input()
c=int(input())
for j in range(1,c+1):
    e=input().split()
    f=['I_love',b[int(e[1])]]
    g='_'.join(f)
    b[int(e[0])]=g
    e=[]
print(b[1])
