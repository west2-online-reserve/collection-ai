class Shopping:
    def __init__(self,商品序号,商品名,单价,总数量,剩余数量):
        self.商品序号 = 商品序号
        self.商品名 = 商品名
        self.单价 = 单价
        self.总数量 = 总数量
        self.剩余数量 = 剩余数量
    def display(self):
        print(f"商品序号: {self.商品序号}, 商品名: {self.商品名}, 单价: {self.单价}, 总数量: {self.总数量}, 剩余数量: {self.剩余数量}")

    def income(self):
        return self.单价 * (self.总数量 - self.剩余数量)

    def setdata(self,商品序号,商品名,单价,总数量,剩余数量):
        self.商品序号 = 商品序号
        self.商品名 = 商品名
        self.单价 = 单价
        self.总数量 = 总数量
        self.剩余数量 = 剩余数量