dictionary  = {}
n = int(input("请输入学生数量："))
for i in range(n):
    sid = int(input("请输入学生学号："))
    name = input("请输入姓名：")
    dictionary[sid] = name
del_keys = []
for sid in dictionary:
    if sid % 2 == 0:
        del_keys.append(sid)
for k in del_keys:
    del dictionary[k]
print(dictionary)