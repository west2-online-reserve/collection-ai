#添加了op机制，1是初始化，2是display,3是计算income,4是修改信息

class Product:
    def __init__(self, id, name, price, total_quantity, remaining_quantity):
        self.__id = id        
        self.__name = name                    
        self.__price = price                  
        self.__total_quantity = total_quantity        
        self.__remaining_quantity = remaining_quantity  

    def display(self):
        print("====== 商品信息 ======")
        print(f"商品序号: {self.__id}")
        print(f"商品名:   {self.__name}")
        print(f"单价:     {self.__price}")
        print(f"总数量:   {self.__total_quantity}")
        print(f"剩余数量: {self.__remaining_quantity}")
        print("======================")

    def income(self):
        sold_quantity = self.__total_quantity - self.__remaining_quantity
        total_income = sold_quantity * self.__price
        return total_income

    def setdata(self, id, name, price, total_quantity, remaining_quantity):
        self.__id = id
        self.__name = name
        self.__price = price
        self.__total_quantity = total_quantity
        self.__remaining_quantity = remaining_quantity

while True:
    op = input("请输入操作编号 (0-4): ").strip()

    if op == '1':
        print("\n--- 请输入商品信息（格式：序号 名称 单价 总数 剩余数）---")
        data = input("例如 (101 苹果 5.5 100 80): ").split()
        p = Product(int(data[0]), data[1], float(data[2]), int(data[3]), int(data[4]))
            
    elif op == '2':
        p.display()

    elif op == '3':
        print(f"\n已售出商品的总价值为: {p.income()} 元\n")

    elif op == '4':
        data = input("例如 (101 苹果 6.0 100 50): ").split()
        p.setdata(int(data[0]), data[1], float(data[2]), int(data[3]), int(data[4]))

    elif op == '0':
        print("\n程序已退出")
        break
