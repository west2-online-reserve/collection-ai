
class MyZoo:
    #ani = None并加上判断，否则无参会报错
    def __init__(self, ani = None):
        print("My Zoo!")
        self.animals = ani if ani is not None else {}

    def __str__(self):
        #result = f"动物名称：{self.animals.keys()}，动物数量：{self.animals.values()}"
        result = ""
        for key, value in self.animals.items():
            result += f"名称：{key}，数量：{value}\n"
        return result

    def __eq__(self1, self2):
        return set(self1.animals.keys()) == set(self2.animals.keys())

    def __len__(self):
        return sum(self.animals.values())

def main():
    myzooo1 = MyZoo({'pig': 5, 'monkey': 10})
    myzooo2 = MyZoo()
    print(myzooo1, end = '')
    print(myzooo1 == myzooo2)
    print(len(myzooo1))

if __name__ == "__main__":
    main()
