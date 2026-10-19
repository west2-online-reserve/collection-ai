# 函数
def record(l):
    c = 0
    d = {}
    l = "".join(str(i) for i in l)
    for i in range(1, 10):
        c = 0
        for j in l:
            if i == int(j):
                c += 1
        d[i] = c
    print(d)


# 商品
class obj:
    def __init__(self, rank, name, money, total, rest):
        self.__rank = rank
        self.__name = name
        self.__money = money
        self.__total = total
        self.__rest = rest

    def display(self):
        d = {
            "序号": self.__rank,
            "名字": self.__name,
            "单价": self.__money,
            "总量": self.__total,
            "剩余数量": self.__rest,
        }
        print(d)

    def income(self):
        v = self.__money * (self.__total - self.__rest)
        print("赚的钱：", v)

    def setdata(self, rank, name, money, total, rest):
        self.__rank = rank
        self.__name = name
        self.__money = money
        self.__total = total
        self.__rest = rest