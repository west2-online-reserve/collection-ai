d = {"001":"zhangsan", "010":"lisi", "901":"wangwu"}

#遍历字典的同时修改字典会报错
for key in list(d.keys()) :
    num = int(key)
    if num % 2 == 0 :
        d.pop(key)

print(d)