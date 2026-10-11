class Product:
    def __init__(self,pid,name,price,total,remain):
        self.__pid=pid
        self.__name=name
        self.__price=price
        self.__total=total
        self.__remain=remain

    def display(self):
        print(f"序号: {self.__pid}")
        print(f"名称: {self.__name}")
        print(f"单价: {self.__price}")
        print(f"总数量: {self.__total}")
        print(f"剩余数量: {self.__remain}")
        print(f"已售出: {self.__total-self.__remain}")

    def income(self):
        sold = self.__total-self.__remain
        return sold*self.__price

    def setdata(self,pid=None,name=None,price=None,total=None,remain=None):
        if pid is not None:
            self.__pid=pid
        if name is not None:
            self.__name=name
        if price is not None:
            self.__price=price
        if total is not None:
            self.__total=total
        if remain is not None:
            self.__remain=remain