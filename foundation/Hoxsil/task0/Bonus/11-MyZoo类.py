class MyZoo(object):
    def __init__(self, dict1: dict[str, int] | None = None) -> None:
        if dict1 is None:
            self.animals: dict[str, int] = {}
        else:
            self.animals: dict[str, int] = dict1
        print("My Zoo!")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, My Zoo):
            return False
        return self.animals.keys() == other.animals.keys()

    def print(self) -> None:
        for name in self.animals:
            print(f"{name}的数量为：{self.animals[name]}")

    def len(self) -> None:
        print(sum(self.animals.values()))