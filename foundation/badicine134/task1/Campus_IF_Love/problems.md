# 写山寨Galgame时遇到的问题

阅读代码框架时费了蛮大劲

主要是方法调用的逻辑没学明白，以及定义实例属性（尤其是``self.``代表当前实例，还有``__init__``初始化）和创建实例对象不熟练

现在有点理解了

```Py
    def __init__(self):
        self.characters = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无")
        }
        self.current_target = self.characters["None"]
```

这段的写法也是学习到了
