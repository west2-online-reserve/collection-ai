def count(a):
    b={}
    for i in a:
        try:
            j = int(i)
        except ValueError:
            b = ("列表不是纯数字")
            break
        if j in b:
            b[j] += 1
        else:
            b[j] = 1
    return b
#测试
num_li = input().split()
print(count(num_li))