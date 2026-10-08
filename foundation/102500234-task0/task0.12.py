class MyZoo:
    def __init__(self, dic=None):
        print("My Zoo!")
        if dic is None:
            self.animals = {}
        else:
            self.animals = dic.copy()

    def __str__(self):
        res = []
        for k, v in self.animals.items():
            res.append(f"{k}:{v}")
        return " ".join(res)

    def __eq__(self, other):
        if not isinstance(other, MyZoo):
            return False
        return set(self.animals.keys()) == set(other.animals.keys())

    def __len__(self):
        return sum(self.animals.values())
