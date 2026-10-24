class Product:
    def __init__(self,num,name,price,total_quanity,rest_quanity):
            self.num = num
            self.name = name
            self.price = price
            self.total_quanity = total_quanity
            self.rest_quanity = rest_quanity

    def display(self):
          print(f"序号:{self.num};品名:{self.name};单价:{self.price};总数量:{self.total_quanity};剩余数量:{self.rest_quanity}")

    def income(self):
          a = (self.price)*(self.total_quanity-self.rest_quanity)
          print(f"{self.name}已经售出{a}的价值")

    def setdata(self,num,name,price,total_quanity,rest_quanity):
        self.num = num
        self.name = name
        self.price = price
        self.total_quanity = total_quanity
        self.rest_quanity = rest_quanity

#测试
sb = Product(0,"陈帅",-100,1,0)
sb.display()
sb.income()
          