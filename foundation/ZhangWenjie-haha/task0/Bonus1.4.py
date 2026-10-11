a = input().split()
num = []

for x in a:
    if all('0' <= c <= '9' for c in x):
        num.append(int(x))

num.sort()

print(num)
