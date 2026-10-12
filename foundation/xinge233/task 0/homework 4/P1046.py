# h=[map(int,input().split())] map导致列表数据异常,学习了map函数
h = list(map(int, input().split()))  # 通过list转换map迭代器
taotao = int(input())
t = taotao + 30
num = 0
for a in range(len(h)):
    if t >= h[a]:
        num += 1
# print(h)
print(num)
