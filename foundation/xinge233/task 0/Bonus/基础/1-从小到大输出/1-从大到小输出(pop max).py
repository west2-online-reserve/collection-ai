a, b, c = map(int, input().split())
lst = [a, b, c]
result = []
while len(lst) > 0:
    result.append(lst.pop(lst.index(max(lst))))
print(*result)
# 这里加了*后，输出变成了列表的内容，即字符串
# 未加*则会输出列表本身，结果会被[]包裹
