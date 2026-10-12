n = int(input())
names = list()
for i in range(0,n):
    names.append(input())
m = int(input())
for i in range(0,m):
    u,v=map(int,input().split())
    ##print(u," ",v)
    names[u-1] = "I_Love_" + names[v-1]
##print(*names)
print(names[0])
