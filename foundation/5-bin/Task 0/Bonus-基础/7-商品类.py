class Product :
    """
    商品类
    """
    __number = 0
    __name = ''
    __price = 0
    __total = 0
    __leftnum = 0

    def __init__(self, num, nam, pri, tot, lef) :
        """
        初始化商品信息
        """
        self.__number = num
        self.__name = nam
        self.__price = pri
        self.__total = tot
        self.__leftnum = lef

    def display(self) :
        """
        显示商品信息
        """
        print(self.__number, self.__name, self.__price,
              self.__total, self.__leftnum ,end = ' ')
        print('')

    def income(self) :
        """
        计算商品销售收入
        """
        return (self.__total-self.__leftnum)*self.__price

    def setdata(self, num=None, nam=None, pri=None,
                tot=None, lef=None):
        """
        修改商品信息
        只修改传入的参数，未传入的参数保持不变
        """
        if num is not None:
            self.__number = num
        if nam is not None:
            self.__name = nam
        if pri is not None:
            self.__price = pri
        if tot is not None:
            self.__total = tot
        if lef is not None:
            self.__leftnum = lef

def main() :
    """
    主函数
    """
    # 创建商品对象
    p = Product(1001, "笔记本", 5999.0, 100, 30)

    # 显示商品信息
    p.display()

    # 计算已售出商品价值
    print("已售出商品价值：", p.income())

    # 修改商品信息
    p.setdata(pri=5499.0, lef=20)

    print("\n修改后：")
    p.display()
    print("已售出商品价值：", p.income())

if __name__ == "__main__" :
    main()
