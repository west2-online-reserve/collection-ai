class MyZoo:
    def __init__(self, animals=None):
        if type(animals) == dict:
            self.animals = animals
        else:
            self.animals = {}
        print("My Zoo!")

    def __str__(self):
        report = []
        for i in self.animals:
            report.append(f"有{self.animals[i]}只{i}")
        return ",".join(report)

    def __eq__(first, second):
        if first.animals.keys() == second.animals.keys():
            return True
        else:
            return False

    def __len__(self):
        total = 0
        for i in self.animals:
            total += self.animals[i]
        print(f"有{total}只动物")
        return total


if __name__ == "__main__":
    test = MyZoo("我偏不传字典")
    print(test.__dict__)
    myzoooo = MyZoo({"pig": 5, "dog": 6})
    print(myzoooo)
    myzoooo1 = MyZoo({"pig": 1})
    myzoooo2 = MyZoo({"pig": 5})
    print(myzoooo1 == myzoooo2)
    myzoooo1 = MyZoo({"bird": 1})
    myzoooo2 = MyZoo({"pig": 5})
    print(myzoooo1 == myzoooo2)
    len(myzoooo)


# return f"""动物名称为{",".join((self.animals).keys())}
# 动物数量为{",".join(map(str,(self.animals).values()))}"""

"""
4. 了解类的魔术方法（Magic Method）。创建类 MyZoo，实现以下功能：
    - 具有字典 `animals`，动物名称作为 key，动物数量作为 value
    - 实例化对象的时候，输出 "My Zoo!"
    - 创建对象的时候可以输入字典进行初始化，若无字典传入，则初始化 `animals` 为空字典

    ```python
    myzoooo = MyZoo({"pig":5,'dog':6}) 
    myzoooo = MyZoo() 
    ```

    - print(myzoooo) 输出 动物名称和数量
    - 比较两个对象是否相等时，只要动物种类一样，就判断相等：

        ```python
        输入：
        myzoooo1 = MyZoo({'pig':1})
        myzoooo2 = MyZoo({'pig':5})
        print(myzoooo1 == myzoooo2)
        输出：
        My Zoo!
        My Zoo!
        True
        ```

    - len(myzoooo) 输出所有动物总数
"""
