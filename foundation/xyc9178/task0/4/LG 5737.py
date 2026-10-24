s = input().split()
a = int(s[0])
b = int(s[1])
sum = 0
out = []
for i in range(a, b+1):
    if i % 4 == 0 and ( i %100 !=0 or i %400 == 0):
        sum += 1
        out.append(str(i))

print(sum)
print(' '.join(out))
