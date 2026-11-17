num=input().split()
height=int(input())
s=0
for n in num:
    if height+30>=int(n):
        s+=1
print(s)