s = input().split()
sb = s.copy()
for i in s:
    try:
        i = int(i)
    except ValueError:
        print(i)
        sb.remove(i)
print(sb)