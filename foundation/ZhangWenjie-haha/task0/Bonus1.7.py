class Product:
    def __init__(self, idx, name, price, total, remain):
        self.__idx = idx
        self.__name = name
        self.__price = price
        self.__total = total
        self.__remain = remain

    def display(self):
        print(f"序号：{self.__idx}")
        print(f"名称：{self.__name}")
        print(f"单价：{self.__price}")
        print(f"总数量：{self.__total}")
        print(f"剩余数量：{self.__remain}")

    def income(self):
        return self.__price * (self.__total - self.__remain)

    def setdata(self, idx=None, name=None, price=None, total=None, remain=None):
        if idx is not None: self.__idx = idx
        if name is not None: self.__name = name
        if price is not None: self.__price = price
        if total is not None: self.__total = total
        if remain is not None: self.__remain = remain


def main():
    pass


if __name__ == '__main__':
    main()
