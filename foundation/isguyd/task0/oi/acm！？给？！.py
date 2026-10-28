total=int(input())
name = []
for i in range(total):
    name.append(input())
total=int(input())
for i in range(total):
    gay1,gay2=map(int,input().split())
    gay1-=1
    gay2-=1
    name[gay1]="I_Love_"+name[gay2]
print(name[0])