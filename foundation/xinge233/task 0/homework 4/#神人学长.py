# 神人学长
n = int(input())
num = []
name = []
for i in range(n):
    num.append(i)
    name.append(input(""))
# print(n)
# print(num)
# print(name)
# for i in name:
# print(i)
m = int(input())
for i in range(m):
    a, b = map(int, input().split())
    name[a - 1] = "I_love_" + name[b - 1]
# print(n)
# print(name)
print(name[0])
