a,b = map(int,input().split())
cnt = 0
t = ()
for i in range(a,b+1):
    if i%4==0 and i%100!=0 or i%400==0:
        cnt = cnt + 1
        t = t + (i,)
print(cnt)
for i in t:
    print(i,end=' ')