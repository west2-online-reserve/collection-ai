class MyZoo:
    def __init__(self, animals: dict[str, int] | None = None):
        self.animals = animals if animals is not None else {}

        print("My Zoo!")

    def __len__(self):
        return sum(self.animals.values())

    def __str__(self):
        return str(self.animals)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MyZoo):
            return False
        return set(self.animals.keys()) == set(other.animals.keys())


myzoooo1 = MyZoo({"pig": 1})
myzoooo2 = MyZoo({"pig": 5})
print(myzoooo1 == myzoooo2)
