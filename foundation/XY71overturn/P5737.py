nianfen=input().split()
run=sum(int(y)%4==0 and (int(y)%100!=0 or int(y)%400==0) for y in range(int(nianfen[0]),int(nianfen[1])+1))
print(run)
for y in range(int(nianfen[0]),int(nianfen[1])+1):
    if int(y)%4==0 and (int(y)%100!=0 or int(y)%400==0):
        print(y,end=" ")