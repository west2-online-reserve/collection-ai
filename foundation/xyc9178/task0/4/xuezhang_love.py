n = int(input())
names=[]    
for i in range(n):
    name = input()
    names.append(name)
print(names)
m = int(input())
for i in range(m):
    s = input().split()
    u = int(s[0])-1
    v = int(s[1])-1
    names[u],names[v] = "I_love_" + names[v],"I_love_" + names[u]
print(names[0])