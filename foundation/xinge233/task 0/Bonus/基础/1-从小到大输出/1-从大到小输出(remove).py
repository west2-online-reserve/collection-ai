a, b, c = map(int, input().split())
lst = [a, b, c]
result = []
while len(lst) > 0:
    result.insert(0, min(lst))
    lst.remove(min(lst))
print(*result)
