class Product:
    def __init__(self, number, name, price, total, left):
        self.__number = number
        self.__name = name
        self.__price = price
        self.__total = total
        self.__left = left

    def display(self):
        print("商品编号：", self.__number)
        print("商品名称：", self.__name)
        print("商品单价：", self.__price)
        print("商品总数量：", self.__total)
        print("剩余数量：", self.__left)

    def income(self):
        sold = self.__total - self.__left
        return sold * self.__price

    def setdata(self, number, name, price, total, left):
        self.__number = number
        self.__name = name
        self.__price = price
        self.__total = total
        self.__left = left


p = Product(1001, "铅笔", 2.5, 100, 60)

p.display()
print("已售出商品价值：", p.income())

p.setdata(1001, "铅笔", 3, 100, 50)

p.display()
print("修改后已售出商品价值：", p.income())
#源自chatGPT