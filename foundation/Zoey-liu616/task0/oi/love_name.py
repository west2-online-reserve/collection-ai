n = int(input())
names = []
for i in range(n):
    name = input()
    names.append(name)
    
m = int(input())
records = []
for i in range(m):  
    u, v = map(int, input().split())
    records.append((u, v))
target = 1
count = 0
for u, v in reversed(records):
    if u == target:
        target = v
        count += 1
answer = "I_love_" * count + names[target - 1]
print(answer)


