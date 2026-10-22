n=int(input())
t=['']
for i in range(1,n+1):
    name=input()
    t.append(name)
m=int(input())
for a in range(1,m+1):
    s=input().split()
    u=int(s[0])
    v=int(s[1])
    name1=t[v]
    name2='I_love_'+name1
    t[u]=name2
print(t[1])

