# P1001
num = input().split()
print(int(num[0])+int(num[1]))

# P1046
apples = list(map(int, input().split()))
taotao = int(input())
max = taotao + 30
count = 0
for height in apples:
    if max >= height:
        count += 1
print(count)

# P5737
x, y = map(int, input().split())
leap_years =[]
for year in range(x,y+1):
    if year % 4 == 0:
        leap_years.append(year)
print(len(leap_years))
print(*leap_years)

# AtCoder ARC017A
n = int(input())
points = 0
if n == 1:
    points += 1
elif n == 2:
    points += 0
elif n % 2 == 0:
    points += 1
for i in range(3,int(n)+1,2):
    if n % i == 0:
        points += 1
else:
    points += 0
if points <= 1:
    print('参加')
else:
    print('不参加')

# 5.
n = int(input())
names = []
for _ in range(n):
    names.append(input())
m = int(input())
for _ in range(m):
    u, v = map(int, input().split())
    names[u - 1] = "I_love_" + names[v - 1]
print(names[0])

