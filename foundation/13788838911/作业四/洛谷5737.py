x,y=map(int,input().split())
lst=[]
cnt=0
for i in range(x,y+1):
    if((i%4==0 and i%100!=0)or(i%400==0)):
        cnt+=1
        lst.append(i)
print(cnt)
print(' '.join(map(str,lst)))