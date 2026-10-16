n = int(input())
name = [None]
for i in range(n):
    name.append(input())
m = int(input())
for i in range(m):
    a, b = map(int, input().split())
    name[a] = "I_love_" + name[b]
print(name[1])