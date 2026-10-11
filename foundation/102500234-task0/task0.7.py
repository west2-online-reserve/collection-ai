#设计⼀个商品类，它具有的私有数据成员是商品序号、商品名、单价、总数量和剩余数量。
#具有的 公有成员函数是：初始化商品信息的构造函数 __init__，显示商品信息的函数 display，
#计算已售出 商品价值 income，修改商品信息的函数 setdata
class Shangpin:
    def __init__(self, num, name, price, all_count, remain):
        self.__num = num              # 商品序号
        self.__name = name            # 商品名
        self.__price = price          # 单价
        self.__all_count = all_count  # 总数量
        self.__remain = remain        # 剩余数量
    def display(self):
        print(self.__name)
        print(self.__num)
        print(self.__price)
        print(self.__all_count)
        print(self.__remain)
    def income(self):
        self.__income1 = (self.__all_count - self.__remain) * self.__price
        print(self.__income1)
    def setdata(self,which1,what1):
        #双下划线属性 self.__num 在类内部会被 Python 改名（name mangling）成 _Shangpin__num。
        #所以用 setattr 时，必须写改名后的真实名字。
        setattr(self,which1,what1) 

