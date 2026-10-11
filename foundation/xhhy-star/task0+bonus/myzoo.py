class MyZoo:
    def __init__(self, animals=None):
        print("My zoo!")
        if animals is None:
            self.animals = {}
        else:
            self.animals = animals

    def __str__(self):
        result_list = []
        for animal, count in self.animals:
            result_list.append(f"{animal}: {count}")
        return "\n".join(result_list)

    def __eq__(self, other):
        if not isinstance(other, MyZoo):
            return False
        return set(self.animals.keys()) == set(other.animals.keys())

    def __len__(self):
        return sum(self.animals.values())

# 测试使用
myzoooo1 = MyZoo({'pig':1})
myzoooo2 = MyZoo({'pig':5})
print(myzoooo1 == myzoooo2)