# 输入⼀个列表（list）
# 列表中含有字符串和整数，删除其中的字符串元素
# 然后把剩下的整数升序排序，输出列表

s = input().split()
num = []
for i in s:
    try:
        temp = int(i)
        num.append(temp)
    except:
        continue
num.sort()
print(num)