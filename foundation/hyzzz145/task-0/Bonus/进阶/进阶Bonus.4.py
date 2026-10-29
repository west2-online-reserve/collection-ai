class MyZoo():
    def __init__(self,dic = None):
        print("My Zoo!")
        if dic == None:
            self.animal = {}
        else:
            self.animal = dic
    def __str__(self):
        if self.animal:
            lines = [f"{name}: {count}" for name, count in self.animal.items()]
            return "\n".join(lines)
        else:
            return("没有任何动物")
    def __eq__(self,other):
        if not isinstance(other,MyZoo):
            print(f"{other} 不在 MyZoo 类里")
            return None
        else:
            return (self.animal.keys() == other.animal.keys())
    def __len__(self):
        return sum(count for count in self.animal.values())
    

myzooo = MyZoo()
myzooo2 = MyZoo({'pig':3,'cat':3})
print(myzooo)
print(len(myzooo))
print(myzooo == myzooo2)

        
    
    

    