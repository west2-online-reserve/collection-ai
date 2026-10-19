a = input().split()
b = []
for i in a:
    try:
        b.append(int(i))
    except ValueError:
        pass
b.sort()
print(b)
