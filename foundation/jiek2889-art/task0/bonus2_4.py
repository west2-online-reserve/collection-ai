class MyZoo:
    def __init__(self, animals=None):
        print("My Zoo!")
        self.animals = {}
        if animals != None:
            self.animals = dict(animals)

    def __str__(self):
        if not self.animals:  # reference from deepseek  not包括了空字典和NONE
            return "无动物"
        total = []
        for a, b in self.animals.items():
            total.append(f"{a}有{b}只")
        return ",".join(total)

    def __eq__(self, other):
        if not isinstance(other, MyZoo):
            return None
        return set(self.animals) == set(self.animals)

    def __len__(self):
        sums = sum(self.animals.values())
        return sums


myzoooo1 = MyZoo({"pig": 1})
myzoooo2 = MyZoo({"pig": 5})
print(myzoooo1 == myzoooo2)
print(len(myzoooo2))
