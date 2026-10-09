s = input()
check = [False] * len(s)

for x in range(0, len(s) - 1):
    if s[x] + s[x + 1] == 'ol':
        check[x + 1] =  True
ans = ''
x = -1
while x >= -len(s):
    if check[x]:
        ans += 'uzf'
        x -= 2
        continue
    else:
        ans += s[x]
    x -= 1

print(ans)

