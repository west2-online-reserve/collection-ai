L = input().split()

i = 0
while i < len(L):
    if not L[i].isdigit():
        L.remove(L[i])
    else:
        i += 1
L.sort(key=int)
print(*L)
