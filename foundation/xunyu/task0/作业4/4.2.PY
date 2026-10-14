height=list(map(int,input().split()))
hr=int(input())
m_h=hr+30
count=0
for h in height:
    if h<=m_h:
        count+=1
print(count)

