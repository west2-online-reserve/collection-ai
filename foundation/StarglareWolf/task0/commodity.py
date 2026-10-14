class Commodity:
    def __init__(self, num, name, price, volume, remain):
        self.__num = num
        self.__name = name
        self.__price = price
        self.__volume = volume
        self.__remain = remain

    def display(self):
        print(f"{self.__num}:{self.__name},售价:{self.__price},库存:{self.__remain},总量:{self.__volume}")

    def income(self):
        money = self.__price * (self.__volume - self.__remain)

    def setdata(self, num = None, name = None, price = None, volume = None, remain = None):
        if num != None: self.__num = num
        if name != None: self.__price = price
        if volume != None: self.__volume = volume
        if remain != None: self.__remain = remain
        if price != None: self.__price = price