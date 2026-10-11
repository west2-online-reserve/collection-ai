#创建一个函数，这个函数可以统计一个只有数字的列表中各个数字出现的次数，通过字典方式返回
def count (a):
    s = {}
    for i in a:
        if int(i) in s:
            s[i] = s[i] +1
        else:
            s[i] = 1
    print(s)
a = list(input().split())
count(a)




