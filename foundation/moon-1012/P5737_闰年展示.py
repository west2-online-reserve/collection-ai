x,y=map(int,input().split())
cnt=0
year=[]
for i in range(x,y+1):
    if(i%4==0 and i%100!=0) or (i%400==0):
        year.append(i)
        cnt+=1
print(cnt)
if year:
    print(" ".join(map(str,year)))
