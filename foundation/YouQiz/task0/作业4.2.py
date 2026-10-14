# 读入10个苹果高度，转为列表
data = input().split()
apple = list(map(int, data[:10]))
# 获取最大高度
h = int(input())
max_h = h + 30

count = 0
for i in apple:
    if max_h >= i:
        count += 1

print(count)