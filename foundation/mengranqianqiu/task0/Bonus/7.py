class Product:
    def __init__(self, num, name, price, total, remain):
        self.__num = num
        self.__name = name
        self.__price = price
        self.__total = total
        self.__remain = remain
    def display(self):
        print(f"商品编号: {self.__num}")
        print(f"商品名称: {self.__name}")
        print(f"商品价格: {self.__price}")
        print(f"商品总量: {self.__total}")
        print(f"商品剩余量: {self.__remain}")
    def income(self):
        return self.__price * (self.__total - self.__remain)
    def setdata(self, num=None, name=None, price=None, total=None, remain=None):
        if num is not None:
            self.__num = num
        if name is not None:
            self.__name = name
        if price is not None:
            self.__price = price
        if total is not None:
            self.__total = total
        if remain is not None:
            self.__remain = remain
def main():
    num = input("请输入商品编号: ")
    name = input("请输入商品名称: ")
    price = float(input("请输入商品价格: "))
    total = int(input("请输入商品总量: "))
    remain = int(input("请输入商品剩余量: "))
    product = Product(num, name, price, total, remain)
    product.display()
    print(f"商品收入: {product.income()}")
if __name__ == "__main__":
    main()
