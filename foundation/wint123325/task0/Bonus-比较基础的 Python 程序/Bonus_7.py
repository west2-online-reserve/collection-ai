class Good:
    def __init__(self,order,name,price,tn,rn):
        self.__order = order
        self.__name = name
        self.__price = price
        self.__tn = tn
        self.__rn = rn
    def display(self):
        print("order:%s"%self.__order)
        print("name:%s"%self.__name)
        print("price:%.2f"%self.__price)
        print("tn:%.2f"%self.__tn)
        print("rn:%.2f"%self.__rn)
    def income(self):
        print("income:%.2f"%((self.__tn-self.__rn)*self.__price))
    def setdata(self,new_order,new_name,new_price,new_tn,new_rn):
        self.__order =new_order
        self.__name =new_name
        self.__price =new_price
        self.__tn =new_tn
        self.__rn =new_rn
#a=Good("A1","wow",100,50,30)
#a.display()
#a.income()
#a.setdata("A2","wow PLUS",688,30,20)
#a.display()
#a.income()
