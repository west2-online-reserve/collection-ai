class Goods:
    def __init__(self,id,name,price,allcount,count):
        self.__id = id
        self.__name = name
        self.__price = price
        self.__allcount = allcount
        self.__count = count

    def display(self):
        income = self.__price * (self.__allcount - self.__count)
        print(f"{self.__id}:{self.__name} 已售出{income}元")

    def setdata(self):
        print(f"\n开始修改 {self.__name} 的数据：")
        self.__price = float(input(f"{self.__name} price:"))
        self.__allcount = int(input(f"{self.__name} allcount:"))
        self.__count = int(input(f"{self.__name} count:"))

apple = Goods(1,"apple",1,40,21)
apple.display()

banana = Goods(2,"banana",3,12,4)
banana.display()

apple.setdata()
apple.display()