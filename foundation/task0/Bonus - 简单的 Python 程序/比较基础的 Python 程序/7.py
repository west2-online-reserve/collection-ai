# 设计⼀个商品类，
# 它具有的私有数据成员是商品序号、商品名、单价、总数量和剩余数量。
# 具有的 公有成员函数是：
# 初始化商品信息的构造函数 __init__，
# 显示商品信息的函数 display，
# 计算已售出 商品价值 income，
# 修改商品信息的函数 setdata

class good:
    def __init__(self,num,name,price,count,remain):
        self.__num = num
        self.__name = name
        self.__price = price
        self.__count = count
        self.__remain = remain

    def display(self):
        print(f"商品序号:{self.__num}")
        print(f"商品名:{self.__name}")
        print(f"单价:{self.__price}")
        print(f"总数量:{self.__count}")
        print(f"剩余数量:{self.__remain}")


    def income(self):
        return self.__price*(self.__count - self.__remain)

    def setdata(self,num=None,name=None,price=None,count=None,remain=None):
        if num!=None:
            self.__num = num
        if name!=None:
            self.__name = name
        if price!=None:
            self.__price = price
        if count!=None:
            self.__count = count
        if remain!=None:
            self.__remain = remain