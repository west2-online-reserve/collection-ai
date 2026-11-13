class MyZoo:
    def __init__(self, animal):
        self.__animal = animal

    def __str__(self):
        result = ""
        for key, value in self.__animal.items():
            result += f"{key}: {value}\t"
        return result

    def __eq__(self, other):
        if not isinstance(other, MyZoo):
            return NotImplemented
        return set(self.__animal.keys()) == set(other.__animal.keys())

    def __len__(self):
        return sum(self.__animal.values())

myzoo = MyZoo({"pig":5,"hen":6})
print(myzoo)
print(len(myzoo))
myzoo2 = MyZoo({"pig":5})
print(myzoo2 == myzoo)