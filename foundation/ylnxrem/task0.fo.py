#洛谷P1006
a=input().split()
print(int(a[0])+int(a[1]))




#洛谷P1046
b=input().split()
c=int(input())
c+=30  
d,e=0,0
while d<10:
    if int(b[d])<=c:
        e+=1
    d+=1
print(e)



#洛谷P5737
x,y=map(int,input().split())
n=0
z=[]
for i in range(x,y+1):
    if (i%4==0 and i%100!=0)or(i%400==0):
        n+=1
        z.append(i)    
print(n)
print(z)


#AT
N=int(input())
i=2
while i<int(N**0.5)+1:
    if N%i==0:
        print('no')
        break
    i+=1
else:
    print('yes')



#学长的爱
n=int(input())
names=['']*n
for i in range(n):
    names[i]=input()
m=int(input())    
for t in range(m):
    u,v=map(int,input().split())
    names[u-1]='I_love_'+names[v-1]
print(names[0])

