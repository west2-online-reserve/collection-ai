class good:
    def __init__(self, no, name, price, total):
        self.__no = no
        self.__name = name
        self.__price = price
        self.__total = total
        self.__remain = total

    def display(self):
        print("商品序号：", self.__no)
        print("商品名：  ", self.__name)
        print("单价：    ", self.__price)
        print("总数量：  ", self.__total)
        print("剩余数量：", self.__remain)

    def income(self):
        sold = self.__total - self.__remain
        return sold * self.__price

    def setdata(self, name=None, price=None, total=None, remain=None):
        if name != None:
            self.__name = name
        if price!= None:
            self.__price = price
        if total != None:
            self.__total = total
        if remain != None:
            self.__remain = remain

g = good(1, "钢笔", 15.5, 100)
g.display()
print("当前销售额：", g.income())
print()
g.setdata(remain=70)
g.display()
print("当前销售额", g.income())
