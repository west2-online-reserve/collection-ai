num = int(input())
name = []
for i in range(num):
    name.append(input())
count = int(input())
for i in range(count):
    a,b = map(int, input().split())
    name[a-1] = "I_love_" + name[b-1]
print(name[0])