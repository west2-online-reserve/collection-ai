class MyZoo:
    
    def __init__(self,ani = None):
        if ani == None:
            self.aninmals = {}
        else:
            self.aninmals = ani
        print("My Zoo")

    def len(self):
        a = 0
        for i in self.aninmals.values():
            a += i
        print(a)

#测试
myzoo1 = MyZoo()
print(myzoo1.aninmals)
myzoo2 = MyZoo({"big":5,"陈帅":1})
print(myzoo2.aninmals)
myzoo2.len()