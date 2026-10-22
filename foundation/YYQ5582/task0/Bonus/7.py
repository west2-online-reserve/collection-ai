class goods:
    def __init__(self,num,name,price,total_number,rest_number):
        self.__num=num
        self.__name=name
        self.__price=price
        self.__total_nummber=total_number
        self.__rest_number=rest_number

    def display(self):
        print("商品序号：{self.__num}")
        print("商品名：{self.__name}")
        print("单价：{self.__price}")
        print("总数量：{self.__total_number}")
        print("剩余数量：{self.__rest_number}")

    def income(self):
        total_income=self.__price*(self.__total_nummber-self.__rest_number)
        return total_income

    def setdata(self,num=None,name=None, price=None, total_num=None, rest_num=None):
        if num is not None:
            self._num = num
        if name is not None:
            self._name = name
        if price is not None:
            self._price = price
        if total_num is not None:
            self._total_num = total_num
        if rest_num is not None:
            self._rest_num = rest_num
        
        