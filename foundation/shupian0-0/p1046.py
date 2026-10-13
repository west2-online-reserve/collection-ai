s = input()
numbers = s.split()
apples = []
for x in numbers:
    apples.append(int(x))

reach = int (input())
highest = reach + 30

count = 0
for h in apples:
    if h <= highest:
        count = count + 1
print(count)

