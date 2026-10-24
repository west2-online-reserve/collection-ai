d = {1:"张三",2:"李四",3:"牛五",4:"晨帅",5:"蒋家林",6:"房子豪"}
db = d.copy()
for key in d:
    if key %2 == 0:
        del db[key]
print(db)