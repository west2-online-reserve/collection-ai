def check(i):
    if (i % 4 == 0 and i % 100 != 0) or i % 400 == 0:
        return True
    else:
        return False


s = input()
start, end = map(int, s.split())
sum = 0
ans = []
for i in range(start, end + 1):
    if check(i):
        sum += 1
        ans.append(i)
print(sum)
for i in ans:
    print(i, end=" ")
