# Object and Class

## Class

类（class）：是一类事物的抽象模板，定义这类事物共有的属性和行为。

对象=属性加方法

```python
class Animal:
    head = 1
    eyes = 2
    def run():
        pass
dog = Animal()  #实例化一个对象
dog.leg#->4
dog.run  #调用方法
dog.leg = 3  #修改对象属性
dir(dog)  #查看对象的属性
```

### 继承

```python
class B(A):  #类B继承自类A拥有与类A相同的属性与方法
    def method_A():
        pass  #若类B中有A中的方法，则将其覆写
isinstance(b , B) # 实例是否在类中
```

#### 钻石继承与MRO

```python
class A:
    pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):  # 钻石继承
    pass
```

这样的话A会初始化两次，正确做法：

```py
class A:
    def __init__(self):
        print("A.__init__")
        super().__init__()

class B(A):
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A):
    def __init__(self):
        print("C.__init__")
        super().__init__()

class D(B, C):
    def __init__(self):
        print("D.__init__")
        super().__init__()

D()
```

顺序：
D -> B -> C -> A -> object

super().m()从当前类之后沿 MRO 找 m 类体内方法

## 构造函数

```py
class C:
    def __init__(self,x,y):
        self.x = x  #接受传入的x参数作为属性x
        self.y = y
    def add (self):
        return self.x + self.y
c = C (2,3)
c.add()
```

self参数的具体作用：

将实例化对象与方法绑定，相当于传入该类的实例

## 组合

```py
class Garden:
    c = Cat()
    d = Dog()
    def say(self):
        self.c = say()
        self.d = say()
```
类中方法共享，属性是各自的，互不干扰

## 多态

```py
class Shape:
    def __init__(self,name):
        self.name = name
class Square(Shape):
    def __init__(self,length):
        super().__init__("正方形")
        self.length = length
class Circle(Shape):
    def __init__(self,radius):
        super().__init__("圆形")
        self.radius = radius
```

name 为一个多态接口

## 魔法方法

形如__init__的方法，由py编译器自动编译

eg:改写python内置的add方法

```py
class S(str):
    def __add__(self,other):
        return len(self) + len(other) #对str类中的__add__方法进行覆写
s1 = S("A")
s2 = S("B")
s1 + s2 # -> 2
```

魔法方法其实就是对接python原生生态的接口

## 属性访问相关

hasasttr(c , " 属性")
getattr(c , "name")
set(c , "name", new)
