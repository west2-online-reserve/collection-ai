aList = []
ans = 0
aList=(list (map(int,input().split())))
height = int(input())+30
for i in range(10):
    if height >= aList[i]:
       ans+=1
print(ans) 