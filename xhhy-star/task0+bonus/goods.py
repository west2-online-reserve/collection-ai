class Product:
    def __init__(self, id, name, price, total, remain):
        self.__id = id  #商品序号
        self.__name = name  #商品名称
        self.__price = price  #商品价格
        self.__total = total  #商品总量
        self.__remain = remain  #商品剩余量

    def display(self):
        """
        显示商品信息
        """
        print(" ==== 商品信息 ====")
        print(f"商品序号: {self.__id}, 商品名称: {self.__name}, 商品价格: {self.__price}, 商品总量: {self.__total}, 商品剩余量: {self.__remain}")

    def income(self):
        """
        计算商品销售收入
        :return: 销售收入
        """
        sold = self.__total - self.__remain  # 已售数量
        value = self.__price * sold  # 销售收入
        return value

    def setdata(self, new_price, new_total, new_remain):
        """
        设置商品价格、总量和剩余量
        :param new_price: 新价格
        :param new_total: 新总量
        :param new_remain: 新剩余量
        """
        self.__price = new_price
        self.__total = new_total
        self.__remain = new_remain
        print("商品信息已更新。")


    
# 测试使用

p1 = Product(1, "柚子", 3.5, 100, 80)
p1.display()
print(f"销售收入: {p1.income()}")
p1.setdata(4.0, 120, 90)
p1.display()