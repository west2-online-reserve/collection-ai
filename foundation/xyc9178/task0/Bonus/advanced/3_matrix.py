a = [[0 for _ in range(5)] for _ in range(10)]#里面行外面列

print(a)
#做成函数试试有没有效果
def turn(x):
    #print(len(x[0]))
    #print(len(x))
    y = [[int(x[j][i]) for j in range(len(x))] for i in range(len(x[0]))]
    #i先锁定列，j再分行搞个列表,用int是因为好像直接循环会直接把x[][]当一个列表直接相加
    #最后会多一个[]，但感觉外面那个应该不能去，不然两个循环粘一起了，不知道有没有更好的办法
    return y

b = turn(a)
print(b)

# c = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
# d = turn(c)
# print(d)