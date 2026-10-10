n = int(input())
t = ()
for i in range(0,n):
    name = input()
    t = t + (name,)
m = int(input())
t1 = ()
t2 = ()
for i in range(0,m):
    a,b = map(int,input().split())
    t1 = t1 + (a,)
    t2 = t2 + (b,)
    if a==1:
        st = i
t3 = (t2[st],)
now = t2[st]
for i in range(st,0,-1):
    if t1[i]==now:
        now = t2[i]
        t3 = t3 + (now,)
for i in range(0,len(t3)):
    print("I_love_",end='')
print(t[t3[0]-1])