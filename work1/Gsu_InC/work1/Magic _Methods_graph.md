# Python魔法方法分类表 
~~快跟我说谢谢ds老师~~
## 一、基础构造与生命周期

| 类别 | 魔法方法 | 触发时机 | 返回值 | 示例 |
|------|----------|----------|--------|------|
| 实例创建 | `__new__(cls, ...)` | 创建实例时（最先调用） | 新实例 | `obj = MyClass()` |
| 初始化 | `__init__(self, ...)` | 实例初始化时 | `None` | `obj = MyClass(arg)` |
| 析构 | `__del__(self)` | 实例销毁时 | `None` | `del obj` |

## 二、字符串与表示

| 类别 | 魔法方法 | 触发时机 | 返回值 | 示例 |
|------|----------|----------|--------|------|
| 字符串表示 | `__str__(self)` | `str()`, `print()` | `str` | `str(obj)` |
| 正式表示 | `__repr__(self)` | `repr()`, 控制台 | `str` | `repr(obj)` |
| 格式化 | `__format__(self, fmt)` | `format()`, `f-string` | `str` | `f"{obj:fmt}"` |
| 字节表示 | `__bytes__(self)` | `bytes()` | `bytes` | `bytes(obj)` |

## 三、算术运算

### 3.1 基本算术运算
| 类别 | 魔法方法 | 运算符 | 示例 | 反向方法 |
|------|----------|--------|------|----------|
| 加法 | `__add__(self, other)` | `+` | `a + b` | `__radd__` |
| 减法 | `__sub__(self, other)` | `-` | `a - b` | `__rsub__` |
| 乘法 | `__mul__(self, other)` | `*` | `a * b` | `__rmul__` |
| 真除法 | `__truediv__(self, other)` | `/` | `a / b` | `__rtruediv__` |
| 地板除 | `__floordiv__(self, other)` | `//` | `a // b` | `__rfloordiv__` |
| 取模 | `__mod__(self, other)` | `%` | `a % b` | `__rmod__` |
| 幂运算 | `__pow__(self, other)` | `**` | `a ** b` | `__rpow__` |

### 3.2 增量赋值运算
| 类别 | 魔法方法 | 运算符 | 示例 |
|------|----------|--------|------|
| 增量加 | `__iadd__(self, other)` | `+=` | `a += b` |
| 增量减 | `__isub__(self, other)` | `-=` | `a -= b` |
| 增量乘 | `__imul__(self, other)` | `*=` | `a *= b` |
| 增量除 | `__itruediv__(self, other)` | `/=` | `a /= b` |
| 增量地板除 | `__ifloordiv__(self, other)` | `//=` | `a //= b` |
| 增量取模 | `__imod__(self, other)` | `%=` | `a %= b` |
| 增量幂 | `__ipow__(self, other)` | `**=` | `a **= b` |

## 四、比较运算

| 类别 | 魔法方法 | 运算符 | 示例 |
|------|----------|--------|------|
| 等于 | `__eq__(self, other)` | `==` | `a == b` |
| 不等于 | `__ne__(self, other)` | `!=` | `a != b` |
| 小于 | `__lt__(self, other)` | `<` | `a < b` |
| 小于等于 | `__le__(self, other)` | `<=` | `a <= b` |
| 大于 | `__gt__(self, other)` | `>` | `a > b` |
| 大于等于 | `__ge__(self, other)` | `>=` | `a >= b` |
| 布尔值 | `__bool__(self)` | `bool()` | `if obj:` |
| 哈希 | `__hash__(self)` | `hash()` | `hash(obj)` |

## 五、类型转换

| 类别 | 魔法方法 | 触发时机 | 返回类型 |
|------|----------|----------|----------|
| 整型转换 | `__int__(self)` | `int()` | `int` |
| 浮点转换 | `__float__(self)` | `float()` | `float` |
| 复数转换 | `__complex__(self)` | `complex()` | `complex` |
| 绝对值 | `__abs__(self)` | `abs()` | 数值类型 |
| 四舍五入 | `__round__(self, n)` | `round()` | 数值类型 |
| 截断 | `__trunc__(self)` | `math.trunc()` | `int` |
| 向下取整 | `__floor__(self)` | `math.floor()` | `int` |
| 向上取整 | `__ceil__(self)` | `math.ceil()` | `int` |
| 索引转换 | `__index__(self)` | 切片索引时 | `int` |

## 六、容器类型方法

### 6.1 序列类型
| 类别 | 魔法方法 | 触发时机 | 示例 |
|------|----------|----------|------|
| 长度 | `__len__(self)` | `len()` | `len(obj)` |
| 获取元素 | `__getitem__(self, key)` | `[]` | `obj[key]` |
| 设置元素 | `__setitem__(self, key, value)` | `[]=` | `obj[key] = value` |
| 删除元素 | `__delitem__(self, key)` | `del` | `del obj[key]` |
| 包含测试 | `__contains__(self, item)` | `in` | `item in obj` |
| 迭代 | `__iter__(self)` | `for` | `for x in obj` |
| 反向迭代 | `__reversed__(self)` | `reversed()` | `reversed(obj)` |

### 6.2 字典类型
| 类别 | 魔法方法 | 触发时机 | 示例 |
|------|----------|----------|------|
| 缺失键处理 | `__missing__(self, key)` | 键不存在时 | `obj[missing_key]` |
| 键集合 | `keys(self)` | `keys()` | `obj.keys()` |
| 值集合 | `values(self)` | `values()` | `obj.values()` |
| 键值对 | `items(self)` | `items()` | `obj.items()` |

## 七、迭代器方法

| 类别 | 魔法方法 | 触发时机 | 示例 |
|------|----------|----------|------|
| 迭代器 | `__iter__(self)` | 迭代开始 | `iter(obj)` |
| 下一个值 | `__next__(self)` | 获取下一项 | `next(obj)` |

## 八、可调用对象

| 类别 | 魔法方法 | 触发时机 | 示例 |
|------|----------|----------|------|
| 函数调用 | `__call__(self, *args, **kwargs)` | `()` | `obj(arg)` |

## 九、属性访问

| 类别 | 魔法方法 | 触发时机 | 示例 |
|------|----------|----------|------|
| 获取属性 | `__getattr__(self, name)` | 属性不存在时 | `obj.missing` |
| 设置属性 | `__setattr__(self, name, value)` | 设置属性时 | `obj.attr = value` |
| 删除属性 | `__delattr__(self, name)` | 删除属性时 | `del obj.attr` |
| 属性访问 | `__getattribute__(self, name)` | 所有属性访问时 | `obj.attr` |
| 属性列表 | `__dir__(self)` | `dir()` | `dir(obj)` |

## 十、描述符方法

| 类别 | 魔法方法 | 触发时机 | 说明 |
|------|----------|----------|------|
| 获取值 | `__get__(self, instance, owner)` | 获取属性时 | 描述符协议 |
| 设置值 | `__set__(self, instance, value)` | 设置属性时 | 描述符协议 |
| 删除值 | `__delete__(self, instance)` | 删除属性时 | 描述符协议 |

## 十一、上下文管理器

| 类别 | 魔法方法 | 触发时机 | 示例 |
|------|----------|----------|------|
| 进入上下文 | `__enter__(self)` | `with`开始时 | `with obj:` |
| 退出上下文 | `__exit__(self, exc_type, exc_val, exc_tb)` | `with`结束时 | `with obj:` |
| 异步进入 | `__aenter__(self)` | 异步`with`开始 | `async with obj:` |
| 异步退出 | `__aexit__(self)` | 异步`with`结束 | `async with obj:` |

## 十二、拷贝与序列化

| 类别 | 魔法方法 | 触发时机 | 说明 |
|------|----------|----------|------|
| 浅拷贝 | `__copy__(self)` | `copy.copy()` | 浅拷贝 |
| 深拷贝 | `__deepcopy__(self, memo)` | `copy.deepcopy()` | 深拷贝 |
| 序列化 | `__getstate__(self)` | `pickle.dump()` | 序列化时 |
| 反序列化 | `__setstate__(self, state)` | `pickle.load()` | 反序列化时 |
| 对象大小 | `__sizeof__(self)` | `sys.getsizeof()` | 内存大小 |

## 十三、类与元类

| 类别 | 魔法方法 | 触发时机 | 说明 |
|------|----------|----------|------|
| 子类初始化 | `__init_subclass__(cls)` | 子类创建时 | 类方法 |
| 子类检查 | `__subclasshook__(cls, subclass)` | `issubclass()` | 类方法 |
| 实例检查 | `__instancecheck__(cls, instance)` | `isinstance()` | 类方法 |
| 准备命名空间 | `__prepare__(name, bases, **kwargs)` | 元类创建类时 | 元类方法 |

## 十四、一元操作符

| 类别 | 魔法方法 | 运算符 | 示例 |
|------|----------|--------|------|
| 取负 | `__neg__(self)` | `-` | `-obj` |
| 取正 | `__pos__(self)` | `+` | `+obj` |
| 绝对值 | `__abs__(self)` | `abs()` | `abs(obj)` |
| 按位取反 | `__invert__(self)` | `~` | `~obj` |

## 十五、位运算操作符

| 类别 | 魔法方法 | 运算符 | 示例 | 反向方法 |
|------|----------|--------|------|----------|
| 按位与 | `__and__(self, other)` | `&` | `a & b` | `__rand__` |
| 按位或 | `__or__(self, other)` | `\|` | `a \| b` | `__ror__` |
| 按位异或 | `__xor__(self, other)` | `^` | `a ^ b` | `__rxor__` |
| 左移位 | `__lshift__(self, other)` | `<<` | `a << b` | `__rlshift__` |
| 右移位 | `__rshift__(self, other)` | `>>` | `a >> b` | `__rrshift__` |
| 增量与 | `__iand__(self, other)` | `&=` | `a &= b` | - |
| 增量或 | `__ior__(self, other)` | `\|=` | `a \|= b` | - |
| 增量异或 | `__ixor__(self, other)` | `^=` | `a ^= b` | - |
| 增量左移 | `__ilshift__(self, other)` | `<<=` | `a <<= b` | - |
| 增量右移 | `__irshift__(self, other)` | `>>=` | `a >>= b` | - |

## 十六、数值运算（已废弃但需了解）

| 类别 | 魔法方法 | 运算符 | 说明 |
|------|----------|--------|------|
| 除法 | `__div__(self, other)` | `/` | Python 2除法，Python 3中为`__truediv__` |
| 反向除法 | `__rdiv__(self, other)` | `/` | Python 2反向除法 |
| 增量除法 | `__idiv__(self, other)` | `/=` | Python 2增量除法 |

## 十七、数学运算方法

| 类别 | 魔法方法 | 数学函数 | 说明 |
|------|----------|----------|------|
| 对数 | `__log__(self, base)` | `math.log()` | 对数运算 |
| 指数 | `__exp__(self)` | `math.exp()` | 指数运算 |
| 平方根 | `__sqrt__(self)` | `math.sqrt()` | 平方根运算 |

## 十八、协程与异步

| 类别 | 魔法方法 | 触发时机 | 说明 |
|------|----------|----------|------|
| 异步迭代 | `__aiter__(self)` | 异步迭代开始 | `async for` |
| 异步下一项 | `__anext__(self)` | 异步获取下一项 | `async for` |
| 异步进入 | `__aenter__(self)` | 异步上下文进入 | `async with` |
| 异步退出 | `__aexit__(self)` | 异步上下文退出 | `async with` |
| 等待 | `__await__(self)` | `await`表达式 | `await obj` |

## 使用频率统计

### 高频使用（必须掌握）
1. `__init__` - 构造方法
2. `__str__` - 字符串表示
3. `__repr__` - 正式表示
4. `__len__` - 长度
5. `__getitem__` - 索引访问

### 中频使用（建议掌握）
1. `__add__`, `__sub__`等 - 运算符重载
2. `__eq__`, `__lt__`等 - 比较运算
3. `__call__` - 可调用对象
4. `__enter__`, `__exit__` - 上下文管理
5. `__iter__`, `__next__` - 迭代器

### 低频使用（了解即可）
1. `__getstate__`, `__setstate__` - 序列化
2. `__new__` - 实例创建
3. 元类相关方法
4. 异步相关方法

## 注意事项

1. **一致性原则**：定义了`__eq__`通常需要定义`__hash__`
2. **反向运算**：实现算术运算时考虑反向方法
3. **增量运算**：增量方法应返回`self`
4. **性能优化**：魔法方法会被频繁调用，需注意性能
5. **异常处理**：在适当的方法中处理异常

---

## 快速参考代码模板

```python
class MyClass:
    # 1. 构造与初始化
    def __init__(self, *args, **kwargs):
        pass
    
    # 2. 字符串表示
    def __str__(self):
        return "用户友好的字符串"
    
    def __repr__(self):
        return "可eval的字符串"
    
    # 3. 运算符重载
    def __add__(self, other):
        return self.value + other
    
    def __eq__(self, other):
        return self.value == other.value
    
    # 4. 容器操作
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]
    
    # 5. 可调用对象
    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)
    
    # 6. 上下文管理
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass