n = int(input())
names = []
for i in range(n):
    names.append(input())
m = int(input())
for i in range(m):
    s = input().split()
    u = int(s[0])
    v = int(s[1])
    names[u-1] = "I_love_" + names[v-1]

print(names[0])