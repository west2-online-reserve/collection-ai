zd = {"001":"zhangsan","002":"lisi","003":"wangwu"}
for i in list(zd.keys()):
    if int(i)%2 == 0:
        del zd[i]
print(zd)