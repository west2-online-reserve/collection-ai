x, y = map(int, input().split())

years = []
for year in range(x, y + 1):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        years.append(year)

print(len(years))
print(*years)

# # 同一行空格输出
# # 方法 1: (首选)
# print(*years)  # 解包，默认用空格隔开
# # 方法 2:
# print(' '.join(map(str, years)))
# # 方法 3:
# for i in range(len(years)):
#     if i == len(years)-1:
#         print(years[i])
#     else:
#         print(years[i], end=' ') # 避免行尾多出空格
# # guided by deepseek