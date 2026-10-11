class product:
    def __init__(self,num,name,price,tn,rn):
        self.__num=num
        self.__name=name
        self.__price=price
        self.__tn=tn
        self.__rn=rn

    def display(self):
        print(f"商品序号：{self.__num}")
        print(f"商品名：{self.__name}")
        print(f"单价：{self.__price}")
        print(f"总数量：{self.__tn}")
        print(f"剩余数量：{self.__rn}")

    def income(self):
        return (self.__tn-self.__rn)*self.__price

    def setdata(self,num=None,name=None,price=None,tn=None,rn=None):
        if num is not None:
            self.__num=num
        if name is not None:
            self.__name=name
        if price is not None:
            self.__price=price
        if tn is not None:
            self.__tn=tn
        if rn is not None:
            self.__rn=rn

s=input('请输入：商品序号 商品名 单价 总数量 剩余数量')
num,name,price,tn,rn=s.split()
price=float(price)
tn=int(tn)
rn=int(rn)
p=product(num,name,price,tn,rn)
print("商品信息")
p.display()
print('已售出商品价值',p.income())
print('是否有要修改的数据（YES/NO)YES')
if input()=="YES":
    new_price = input("请输入新单价（不修改直接回车）：")
    new_rn = input("请输入新剩余数量（不修改直接回车）：")

    p.setdata(
        price=float(new_price)  if new_price else None,
        rn=int(new_rn) if new_rn else None,
    )

    print("\n修改后商品信息")
    p.display()
    print('已售出商品价值：', p.income())

        
        
        
        
        


