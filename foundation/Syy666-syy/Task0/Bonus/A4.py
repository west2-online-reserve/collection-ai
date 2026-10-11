class MyZoo:
    def __init__(self,animals=None):
        print("My Zoo!")
        if animals is None:
            self.animals={}
        else:
            self.animals=animals.copy()
    def __str__(self):
        return ','.join(f"{name}:{count}"for name,count in self.animals.items())
    def __eq__(self, other):
        if not isinstance(other,MyZoo):
            return NotImplemented
        return set(self.animals.keys())==set(other.animals.keys())
    def __len__(self):
        return sum(self.animals.values())
myzoooo1 = MyZoo({'pig': 1})
myzoooo2 = MyZoo({'pig': 5})
print(myzoooo1 == myzoooo2)

myzoooo = MyZoo({"pig": 5, 'dog': 6})
print(myzoooo)

myzoooo3 = MyZoo()
print(myzoooo3)
print(len(myzoooo1))