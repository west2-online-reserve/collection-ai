num=int(input())
nn={}
for i in range(1,num+1):
    name=input()
    nn[i]=name
love=int(input())
for i in range(0,love):
    lovea,loveb=map(int,input().split())
    nn[lovea]="I_love_"+nn[loveb]
print(nn[1])