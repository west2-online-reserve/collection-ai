# 创建一个字典（dict）
# 为字典添加几个键为学号，值为姓名元素
# 删除学号尾号为偶数的元素，输出字典

students = {101:'张三',102:'李四',105:'王五',120:'奶龙',143:'坤坤'}
keys = tuple(students.keys())
for key in keys:
    if(key%2==0):
        students.pop(key)
print(students)