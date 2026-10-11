names = []
n = int(input())
for i in range(n):
    name = input()
    names.append(name)
a = int(input())
for g in range(a):
    u,v = map(int,input().split())
    names[u - 1] = "I_love" + names[v-1]
print(f"{names[0]}")
