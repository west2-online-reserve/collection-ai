a = str(input())
for x in range(len(a)):
    if a[x:x+2] == "ol":
        a = a[:x] + "fzu" + a[x+2:]
print(a[::-1])