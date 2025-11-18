# 补充笔记
## 异常
- 基本语法：
```python
try:
    print('try...')
    r = 10 / int('2')
    print('result:', r)
# except可以多个
except ValueError as e: 
    # 异常均派生自基类BaseException
    print('ValueError:', e)
except ZeroDivisionError as e:
    print('ZeroDivisionError:', e)
else: # 无异常执行
    print('no error!')
finally: # finally 一定会执行
    print('finally...')
print('END')

```
## 面向对象
### 继承
- super() 表示父类（实际是按照mro顺序的上一）
- 也可以直接用父类的名字

重写__init__()一定要先调用父类的完成父类的初始化
```py
    def __init__():
        # 父
        super().__init__()
        #... 子类特有初始化
```

## 杂
- 在 Python 中，自定义类型（如自定义类的实例）作为参数传入函数时，传递的是「对象的引用（内存地址）」，而非对象本身的副本—— 这意味着函数内部对对象的修改（如修改属性、调用修改自身的方法）会影响到函数外部的原对象；但如果在函数内部给参数重新赋值（指向新对象），则不会影响原对象。

- TYPE_CHECKING方面的知识很有意思，以及import也是
```py
    import Play # 导入Play模块，里面有Play类
    a=Play() # 这个运行时一定要导入Play才可以
    a.func() # 这个运行可以不用导入Play，这是py神奇的动态：鸭子类型
```

- 在 Python 中，for 循环迭代时直接修改迭代对象（如列表、字典、集合等可变对象），大概率会出现逻辑错误或意外结果，核心原因是：for 循环的迭代过程依赖于对象的“迭代器”（iterator），而修改对象本身会破坏迭代器的遍历逻辑（比如改变元素数量、索引位置）。