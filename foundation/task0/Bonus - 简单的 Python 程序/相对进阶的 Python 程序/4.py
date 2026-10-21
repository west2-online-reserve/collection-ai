class MyZoo():
    def __init__(self, animals=None):
        if animals!=None:
            self.animals = animals
        else:
            self.animals = {}
        print('MyZoo!')

    def __str__(self):
        return str(self.animals)

    def __eq__(self,other):
        return set(self.animals.keys())==set(other.animals.keys())

    def __len__(self):
        return sum(self.animals.values())


myzoooo1 = MyZoo({'pig':1})
myzoooo2 = MyZoo({'pig':5})
print(myzoooo1==myzoooo2)

