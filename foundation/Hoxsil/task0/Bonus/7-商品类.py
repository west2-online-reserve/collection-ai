class commodity:
    num = None
    name = None
    prize = None
    allcount = None
    count = None


    def init(self):
        self.num = None
        self.name = None
        self.prize = None
        self.allcount = None
        self.count = None


    def display(self):
        print(self.num)
        print(self.name)
        print(self.prize)
        print(self.allcount)
        print(self.count)


    def income(self):
        return self.prize*(self.allcount-self.count)

    
    def setdata(self, num1, name1, prize1, allcount1, count1):
            self.num = num1
            self.name = name1
            self.prize = prize1
            self.allcount = allcount1
            self.count = count1


a = commodity()
a.setdata("001", "a", 21, 20, 5) #卖出15，单价21
a.display()
print(a.income())