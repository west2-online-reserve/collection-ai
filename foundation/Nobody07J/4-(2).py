s=input().split()
h=int(input())
num=0
for i in range(1,len(s)+1):
    if int(s[i-1])<=h:
        num=num+1
    elif int(s[i-1])<=(h+30):
        num=num+1
    i=i+1
print(num)