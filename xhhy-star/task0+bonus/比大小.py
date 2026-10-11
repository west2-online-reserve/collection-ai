#方法一：列表排序
x = int(input("请输入x:"))
y = int(input("请输入y:"))
z = int(input("请输入z:"))
lst = [x, y, z]
lst.sort(reverse=True)
print("从大到小排序后的结果为:", lst)

#方法二：通过循环条件判断
a, b, c = x, y, z
if a < b:
    a, b = b, a
if a < c:
    a, c = c, a
if b < c:
    b, c = c, b
print("从大到小排序后的结果为:", [a, b, c])

#方法三：使用内置函数sorted()
x = int(input("请输入x:"))
y = int(input("请输入y:"))
z = int(input("请输入z:"))

lst = sorted([x, y, z], reverse=True)
print("从大到小排序后的结果为:", lst)

#方法四：使用条件表达式
x = int(input("请输入x:"))
y = int(input("请输入y:"))
z = int(input("请输入z:"))

max_num = max(x, y, z)
min_num = min(x, y, z)
mid_num = x + y + z - max_num - min_num
print("从大到小排序后的结果为:", [max_num, mid_num, min_num])

