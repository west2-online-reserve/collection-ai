n = int(input())
name = []
for i in range(n):
    name.append(input())
k = int(input())
front = "I_love_"
for i in range(k):
    num1, num2 = map(int, input().split())
    name[num1 - 1] = front + name[num2 - 1]
print(name[0])
