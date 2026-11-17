n=int(input())
name={}
for i in range(1,n+1):
    name[i]=input()
m=int(input())
for i in range(1,m+1):
    love=input().split()
    love1=int(love[0])
    love2=int(love[1])
    name[love1]='I_love_'+str(name[love2])
print(name[1])
