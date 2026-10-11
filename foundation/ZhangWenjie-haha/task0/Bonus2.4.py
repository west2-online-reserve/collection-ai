# reference from deepseek

class MyZoo:
    def __init__(self, animals=None):
        print("My Zoo!")

        if animals is None:
            self.animals = {}
        else:
            self.animals = dict(animals)

    def __str__(self):
        if not self.animals:
            return "Zoo is empty"
        return ", ".join(f"{name}: {count}" for name, count in self.animals.items())

    def __cmp__(self, other):
        if not isinstance(other, MyZoo):
            return NotImplemented
        return set(self.animals.keys()) == set(other.animals.keys())

    def __len__(self):
        return sum(self.animals.values())


myzoooo = MyZoo({"pig": 5, "dog": 6})
print(myzoooo)

myzoooo = MyZoo()
print(myzoooo)

myzoooo1 = MyZoo({'pig': 1})
myzoooo2 = MyZoo({'pig': 5})
print(myzoooo1 == myzoooo2)

print(len(MyZoo({'pig': 5, 'dog': 6})))
