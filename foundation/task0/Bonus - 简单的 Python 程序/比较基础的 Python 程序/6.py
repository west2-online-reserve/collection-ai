# 创建一个函数
# 这个函数可以统计一个只有数字的列表中各个数字出现的次数
# 通过字典方式返回

def num_count(a):
    res = dict(zip(a,[0]*len(a)))
    for i in a:
        res[i] += 1
    return res


#测试
a = list(map(int,input().split()))
count = num_count(a)
print(count)