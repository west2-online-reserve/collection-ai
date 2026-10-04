#设计⼀个商品类，它具有的私有数据成员是商品序号、商品名、单价、总数量和剩余数量
# 具有的 公有成员函数是：初始化商品信息的构造函数 __init__，显示商品信息的函数 display，计算已售出 商品价值 income，修改商品信息的函数 setdata
class Product:
    def __init__(self,__num:int=0,__name:str="0",__price:float=0,__total:int=0,__remain:int=0):
        self.__num=__num
        self.__name=__name
        self.__price=__price
        self.__total=__total
        self.__remain=__remain
    def display(self):
        print(f"""
商品的序列号为{self.__num}
商品号的商品名为{self.__name}
商品号的单价为{self.__price}元
商品号的总数量为{self.__total}个
商品的剩余数量为{self.__remain}个
""")
    def income(self):
        self.__sold=self.__total-self.__remain
        print(f"""
已售出{self.__sold}份
商品价值为{self.__price*self.__sold}元""")
    def setdata(self):
        try:
            self.__num=int(input("请修改商品序号"))
            self.__name=str(input("请修改商品名"))
            self.__price=float(input("请修改单价"))
            while True:
                self.__total=int(input("请修改总数量"))
                if self.__total<0:
                    print("总数量必须大于0")
                    self.__total=int(input("请修改总数量"))
                else:
                    break
            self.__remain=int(input("请输入剩余数量"))
            while True:
                if self.__remain>self.__total:
                    print("剩余数必须少于总数")
                    self.__remain=int(input("请输入剩余数量"))
                elif self.__remain<0:
                    print("剩余数必须大于零")
                    self.__remain=int(input("请输入剩余数量"))
                else:
                    break
        except ValueError:
            print("请输入正常数字")
            self.setdata()
           
if __name__ == "__main__":
    p=Product()
    p.display()
    p.income()
    p.setdata()
    p.display()
    p.income()