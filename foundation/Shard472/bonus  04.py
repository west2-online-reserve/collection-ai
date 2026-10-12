s = []
for x in input().split():
    try:
        s.append(int(x))
    except:
        ValueError
        pass
s.sort()
print(s)