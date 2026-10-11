dict1 = {"001":"张三", "002":"李四", "003":"徐五", " 004":"王六"}
dict2 = {}
for x in dict1:
    if int(x) % 2 != 0:
        dict2[x] = dict1[x]
dict1=dict2
print(dict1)