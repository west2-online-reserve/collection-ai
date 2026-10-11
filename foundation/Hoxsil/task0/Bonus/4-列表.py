a = input().split()
b = []
for x in a:
    if x.isdigit():
        b.append(int(x))
b.sort()
print(b)