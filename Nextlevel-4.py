class MyZoo:
    def __init__(self, dic = None):
        animals = {}
        if dic != None:
            for i in dic:
                if i not in animals:
                    animals[i] = dic[i]
                else:
                    animals[i] += dic[i]
        self.animals = animals
        print("My Zoo!")

    def __str__(self):
        str=""
        for i in self.animals:
            str += f"{i}: {self.animals[i]}\n"
        return str

    def __eq__(self, other):
        for i in other.animals:
            if i not in self.animals:
                return False
        for j in self.animals:
            if j not in other.animals:
                return False
        return True

    def __len__(self):
        sum=0
        for i in self.animals:
            sum += self.animals[i]
        return sum


myzoooo1 = MyZoo({'pig':1})
myzoooo2 = MyZoo({'pig':5})
print()
print(myzoooo1)
print(myzoooo2)
print(myzoooo1 == myzoooo2)