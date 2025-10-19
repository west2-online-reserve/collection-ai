# **python的数据分析基础入门：**

## 1.numpy库的基本语法及使用：

### 1.基础库函数的实现：

首先引入numpy库 **`import numpy as np` （以后都认为np）**



#### 1.Array函数

```Python
arr = np.array()
#返回一个数组

#示例
arr = np.array([1,2,3,4])

#可以定义多维数组
arr = np.array([1,2],[3,4])

#同时可以传入数组
a = [10,20,30]
arr1 = np.array(a)
```



#### 2.ndim函数

```Python
arr.ndim
#返回多维数组的维度
```



#### 3.shape函数

```Python
arr.shape
#返回数组的个数 返回的类型是元组 如果只有一个元素为（2，）
```

若数组的为 a = [(1,2,3) , (4,5,6)]

输出为（2，3） （元素 ， 小元素）



#### 4.arange函数

```Python
arr = np.arrage(10)
#输出:  (1,2,3,--- , 10)
```

arrange 的使用 （开始，结束，步长）（类似于range）



#### 5.astype函数

```python
arr = ([1,2,3,4,5,6])
arr.astype(np.float)
#将整数转化为小数

```

！！**注意**！！返回的值是一个**新**的数组 并不会改变原有的数组



#### 6.reshape函数

```python
arr = np.arrange(10,20)
#([10,11,12,13,---,19])
arr.reshape(2,5)
#([10,11,12,13,14],[15,16,17,18,19])
```

reshape(行，列)

或者说（两个大元素  ， 一个小元素）

**！！**reshape不会改变原始数组，只会返回一个新的数组



### 2.保存数据的类型：

```python
np.array([10,20,30,40],dtype = np.float)
```

可以通过dtype给数据进行类型的定义

### 3.多维数组的运算：

示例：

```python
arr1 = np.arrange(1,5)
arr2 = np.arrange(6,10)

arr1 + arr2
arr1 - arr2
arr1 * arr2
arr1 * arr2
#所有正常的运算都可以实现 返回一个新的数组 包含运算的结果

10 * arr1
#10作为标量 所以10的每个值直接*10返回结果的值


arr1 > arr2
#返回布尔数组 每个元素进行比较

arr1 > 10
#同样返回布尔数组 每个元素和标量进行比较
```



### 4.基本的索引和切片：

```python
arr = np.array([20,100,50,60])
arr[1]
#返回100

arr[1:3]  #切片操作
#结果返回 array([100 , 50])

#同时可以在切片的同时 对数组进行修改
arr[1:3] = 20
#arr([20,20,20,60]) 改变切片的值会影响原始数组上

```

切片：

```python
#导入numpy包
import numpy as np

#同样的我们可以对多维数组进行切片的操作：
arr =np.array([[10,11,12,13,14],
          [15,16,17,18,19]])
print(arr[1:2,1:3])
#输出
[[16 17]]
```



### 5.布尔索引：

利用布尔索引返回一个新的数组：

```python
#导入numpy包
import numpy as np

#同样的我们可以对多维数组进行布尔索引的操作
arr =np.array([[10,11,12,13,14]])

arr1 = arr[arr >= 11]
print(arr1)
#输出   [11 12 13 14]

arr2 = arr[(arr >= 11) & (arr <= 13)]
print(arr2)
#输出   [11 12 13]

arr3 = np.arange(10,30).reshape(4,5)
arr4 = arr3[[True,False,True,False]]

print(f"原{arr3}")
print(f"更正：{arr4}")
#输出： 
# 原    [[10 11 12 13 14]
#        [15 16 17 18 19]
#        [20 21 22 23 24]
#        [25 26 27 28 29]]

# 更正：[[10 11 12 13 14]
#       [20 21 22 23 24]]
```



### 6.花式索引：

```python
#导入numpy包
import numpy as np

#同样的我们可以对多维数组进行花式索引的操作

arr = np.array([10,100,800,80,90,200])

arr1 = arr[[1,2,4]]
print(arr1)
#输出：[100 800 90]
```

同样的可以在二维数组中进行索引抓取的操作

只需要将[1,2\][3,4]  分别写出他的横纵坐标（索引）即可



### 7.通用函数：（快速点对点函数）



```python
#导入numpy包
import numpy as np

#同样的我们可以对多维数组进行点对点函数

arr = np.array([10,-100,800,-80,90,-200])
a1 = np.abs(arr)
a2 = np.sqrt(4)
print(f"a1 = {a1} , a2 = {a2}")

#输出：   a1 = [ 10 100 800  80  90 200] , a2 = 2.0
```

**常见的点对点函数：**

![image-20241012105305335](C:\Users\Slexy\AppData\Roaming\Typora\typora-user-images\image-20241012105305335.png)

![image-20241012105527789](C:\Users\Slexy\AppData\Roaming\Typora\typora-user-images\image-20241012105527789.png)



二元型：

np.func(array1 , array2)  ->  返回一个数组

![image-20241012105818232](C:\Users\Slexy\AppData\Roaming\Typora\typora-user-images\image-20241012105818232.png)





（注意 大部分的结果是小数（float



### 8.数学和统计操作：

```python
#导入numpy包
import numpy as np

#同样的我们可以对数组进行统计的操作：

arr = np.array([10,-100,800,-80,90,-200])

print(np.sum(arr))   #求和
print(np.max(arr))   #最大值

#输出：
# 520
# 800

#统计相关的函数，也可以直接使用数组进行调用：
arr.sum()
arr.mean()

#对于高维的数组可以传入 asis 参数指定维度轴
arr = np.arange(10,30).reshape(4,5)
print(arr)
#  [[10 11 12 13 14]
#  [15 16 17 18 19]
#  [20 21 22 23 24]
#  [25 26 27 28 29]]

np.sum(arr,axis = 0)   #按行求和
np.sum(arr,axis = 1)   #按列求和

print(np.sum(arr,axis = 0))
print(np.sum(arr,axis = 1))
#输出
# [70 74 78 82 86]
# [ 60  85 110 135]
```

常见的统计方法：

![image-20241012110940045](C:\Users\Slexy\AppData\Roaming\Typora\typora-user-images\image-20241012110940045.png)

![image-20241012111009373](C:\Users\Slexy\AppData\Roaming\Typora\typora-user-images\image-20241012111009373.png)



### 9.排序与唯一值

```python
#导入numpy包
import numpy as np

#同样的我们可以对数组进行排序与唯一值

arr = np.array([10,-100,800,-80,90,-200])
arr1 = np.sort(arr)   #返回一个新的数组 不会对原本发生变化
print(arr1)
#输出[-200 -100  -80   10   90  800]

#降序排列 利用切片进行倒转
arr1[::-1]

#唯一值操作
arr2 = np.unique(arr)
print(arr2)
#可以输出唯一值 而且数组是经过排序的

```



## 2.pandas库的基本语法及其使用：

**pandas主要是处理表格型的数据**

**而numpy主要是处理数据型的数据**

### 1.基础库函数的实现：

首先引入numpy库 **`import pandas as pd` （以后都认为pd）**

### 2.pandas的数据结构：

1. Series
   Seres是一种类似于一维数组的对象，它由一组数据(各种NumPy数据类型)以及一组与之相关的数据标签(即引)组成。仅由一组数据即可产生最简单的Series
2. DataFrame
   DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型(数值、字符串、布尔值等)。
   DataFrame既有行索引也有列索引，它可以被看做由Series组成的字典(共用同一个索引)。DataFrame中的数据是以一个或多个二维块存放的(而不是列表、字典或别的一维数据结构)。

#### 1.Series的使用：

```python
import pandas as pd

a = pd.Series([10,20,30,400,50])
print(a)
#输出：
# 0     10
# 1     20
# 2     30
# 3    400
# 4     50
# dtype: int64
```



可以看到，左边表示index，右边表示对应的vlaue，可以通过value和index属性进行查看

```python
print(pd.Index(a))

# Index([10, 20, 30, 400, 50], dtype='int64')

print(a.values)

# [ 10  20  30 400  50]

print(a.index)

# RangeIndex(start=0, stop=5, step=1)
```



同时比大小同时也可以返回：

```python
print(a > 1)
# 0    True
# 1    True
# 2    True
# 3    True
# 4    True
# dtype: bool
```



同样的可以调用numpy内的函数进行数组的计算：

```python
print(np.sum(a))

# 510
```



在后面指定index可以直接指定前面的序号：

```python
print(pd.Series([100,20,30,50],index = ["a","b","c","d"]))

# a    100
# b     20
# c     30
# d     50
# dtype: int64
```



可以用index对series进行索引：

```python
print(c["a"])
# 100
print(c["0"])
# 100
```

（但是不可以使用-1 获取 因为没有这个索引 如果加了自定义标签 除了可以使用自定义名获取 也可以使用原本的数字进行获取



可以进行赋值的操作

```python
c["a"] = 1000
print(c["a"])
# 1000
```



也可以进行花式索引

```python
c["a","b"] = 1000
print(c["a","b"])
# 1000 1000
```



使用numpy函数或类似的操作，会保留index-value的关系



**数据对齐特色：（按照索引相加)**

```python
import numpy as np
import pandas as pd

c = pd.Series([100,20,30,50],index = ["a","b","c","d"])
d = pd.Series([100,20,30,50],index = ["c","b","a","d"])

print(d+c)

#a    130
# b     40
# c    130
# d    100
# dtype: int64
```



两个Series中运算时，如果索引值不相同，会产生空置   NaN: 空

```python
import numpy as np
import pandas as pd

c = pd.Series([100,20,30,50],index = ["a","b","c","f"])
d = pd.Series([100,20,30,50],index = ["c","b","a","d"])

print(d+c)

# a    130.0
# b     40.0
# c    130.0
# d      NaN
# f      NaN
# dtype: float64
```

空值在参与运算时，结果也是空值，系统的sum，mean，会自动排除空值



Series的index可以直接被修改

```python
import numpy as np
import pandas as pd

d = pd.Series([100,20,30,50],index = ["c","b","a","d"])

c.index = [1,2,3,4]
print(c)

# 1    100
# 2     20
# 3     30
# 4     50
# dtype: int64
```



#### 2.Dateframe的使用：

```python
import numpy as np
import pandas as pd

dic = {"name":["张三","李四","王五","赵六"],"age":[18,19,20,30],"gender":['man','woman','man','women']}

df = pd.DataFrame(dic)
print(df)
```

输出：

> [!IMPORTANT]
>
>    name  age gender
> 0   张三   18    man
> 1   李四   19  woman
> 2   王五   20    man
> 3   赵六   30  women





df.head()函数默认为显示前五行

df.head(a)  a代表几行





df.colums显示的是

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=YTVhNmFmMmMzZGFhMGU2ZGM5NzI1YzIyMmM0Mjk3YjlfVzhKeTRJR0tsbXY4SjRXYUl2UkRjaXJoY2JNU3hGaDJfVG9rZW46QTVZMWJjRDdab3FsUjh4c21WQWNPZEdSbk1iXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

df.index 显示的是：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=NGYzNGYxYzNiOGJjODYzN2E2MmJiZDFmNmJmZmNlOWNfdGFoWFFUYU8xekhWUUZ6VTVTSGRmT2RXNVpYaGVHNThfVG9rZW46WEYzZ2JTWFhQb0RYc2h4b3RMeWNkRFlRbm9iXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

也可以使用df.（字典名进行读取对应的列）

例如：

```Python
import numpy as np
import pandas as pd

dic = {"name":["张三","李四","王五","赵六"],"age":[18,19,20,30],"gender":['man','woman','man','women']}

df = pd.DataFrame(dic)
print(df.age)
```

输出:

> 0    18
>
> 1    19
>
> 2    20
>
> 3    30
>
> Name: age, dtype: int64

也可以通过   **`df.loc[ int ]`**   去访问一行的值

例如我去访问    **`df.loc[ 1 ]`**

输出：

> name         李四
>
> age          19
>
> gender    woman
>
> Name: 1, dtype: object

如果我使用赋值去操作他：

例如`df.age = 20`

输出：

>    name  age gender
>
> 0   张三   20    man
>
> 1   李四   20  woman
>
> 2   王五   20    man
>
> 3   赵六   20  women

会直接操作**所有**的age变成20

同样的如果我们把等号右边变成一个列表那么就会把列表的值按顺序赋给左边

如果我们使用`df["height"] = 1.7 ` 创建一个不存在的键值对

那么输出：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=NmU3MWFhYWM3ZTQ5MGFmMmRmODdkOTM2NTk4Y2U3ZDVfYW51cDVmd2hLTldDTExsMXVpM1FwNnIydDVWOFJIUThfVG9rZW46SlBrTmJDMDNRb0VweVJ4TWFHMWMzVTZWbm9mXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

可以看到新增了一列的`height`的值

***不过如果想要增加新的列，不能使用点的形式，必须使用******`[]`******的形式***

同样的可以使用numpy中的类型转换的方法

```
def.age = df.age.astype(np.float)
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=YzUyYWZjZjZjMjM1NzU5OTBhZDZiZjE5ODBhYWRmZjZfdU1nYXZIWHd6aHM5T3NKN1k5NDRKaUxVUHFSUGo0eDFfVG9rZW46WEZ0WWJKdjhMb1V5aUx4YXV4ZGNaNnVNbnpjXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

***不过注意的是返回的是一个新的数组 所以需要重新进行赋值哦***

#### 3.按轴删除记录

使用.drop的方式删除索引

例如：

```Python
import numpy as np
import pandas as pd

c = pd.Series([100,20,30,50],)
d = pd.Series([100,20,30,50],)

print(c)
c = c.drop(0)
print(c)
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=OGYwNjE5ZmIzZTg3ZTRkODZlNmRmNjU4NGUzYmI3OWFfdkdZSDNGT0o5WGs1UnRTcmFqNnpaZzQyaHFidGpYUk1fVG9rZW46VmNFZ2JKZVNQb0hydEt4cDlaNWN2UXFZbjhiXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

也可以传递多个索引进行删除

```
c = c.drop([1,2])
```

结果：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MjNkYzBhN2Q4Mzk4ODMwMDdmMjFhYzdmMzI2YmQ1MzFfQlRxbHpBY1p1RmVaYWlHTmoyWmFhdVF5bUpOOEhqZFZfVG9rZW46TmxSOWJnWFQzb2xJbHd4Umg0QWNxdVY4blRlXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

对于dateframe，drop方法能够按行或列的方法来删除数据：

***默认是以行进行删除的：***

例如：

```Python
import numpy as np
import pandas as pd

dic = {"name":["张三","李四","王五","赵六"],"age":[18,19,20,30],"gender":['man','woman','man','women']}

df = pd.DataFrame(dic)

print(df)
df = df.drop(2)
print(df)
```

结果： （第二行被删除）

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MmFhZDZjZWY5YWM1OTcwYTY2ZTlmODUwZGQxMjJlYjZfSzFSekJ2QTd0MXcxZGE4RFZmV2drbjEwV25LM2kybk5fVG9rZW46UE1RMmJMYlFnbzdQMTZ4c2dzamMwNVNQblN5XzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

***删除列的时候必须指定轴***

例如：

```
df = df.drop("gender",`*`axis`*` = 1)
```

结果：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=OTJlYTBhZThkZmQ3ZDgyM2ExYzEwOTAxNmE4YjBhZDBfcHU2WHBzQTJQYTlvUnoyck56TjhJdncxRG1Xejd5QkRfVG9rZW46UEJUZ2JWVGJ3b2xEY2N4VW9aaWMwaFpwbjNXXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

小项目 ： 

要求想删除性别为男的序列

操作：

1.获取所有性别为男的索引：

```
a = df.gender == "man"
print(a)
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2ZkNjY3ZjgwMDJkYTUzZTRkNDM5NmI1NDA1NjhkNzhfMlNocndTbFlTZlhkdVlEY2xGS3FPU3Mxd3JRb3lWUlhfVG9rZW46SDZoMGJUZ0NRb1Z1RmV4UDJoYWNFWG1hbjhOXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

2.布尔类型的series也可以对dateframe进行过滤

所以：

**`print(df[df.gender == "man"])`**

就可以得到

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTU1NmRhMTg2ZTEyMmViOGY4ZTdhOGQzYjNlNWYyNWRfSVdvWTRCTlV5Nm5iVjFvTlNjb0EwZG5yWGowQ2U1ZWlfVG9rZW46R21UcWJNdVJGb3BEbHB4Y2xDT2NFb29LbllmXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

如果要删除男的数据

可以先获取男的索引值：***(使用方法：******`a.index`******)***

```
a = df[df.gender == "man"]
print(a.index)
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=Zjg4NzczYWQxMTg5ZGNkYmFiODUxMTJkMjM0YjAxNzdfS05uNlJOVUFUQWdpZWZjd2dZbHQ2ZWtjSDl6d3owQXBfVG9rZW46RTlXeWJZbUlsb0VTTlB4YTZ3VGN2ZmdXblllXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

然后再根据索引值删去数据：

```
df = df.drop(a.index)
print(df)
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MGQ2Y2QxNmY0NGI3NjAyZTg4ZWVmNjZhNDNkOTUzYmVfWnE4cWxCWk41RkNmM1pMRGUwUWNsbnNGeUVUWmxEZjNfVG9rZW46TXJMOWI1QXFKb0hRbXl4VlQxcmNrc20xblFlXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

得到只有woman的数据！

#### 4.索引选择与过滤：

用lable进行切片的时候会发生一个问题：（不同于python的切片包含前不包含尾）而lable切片则是包含头和尾部

例如：

```
print(c)
print(c["a":"c"])
```

输出：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWRlYTUzNTdjZGEwMzdiMmViMTFiMDM2NDBjOWQ5OTFfMmJySzg2SXN1ajRsYmRadkNBZlVOZkczc3dNS0NnQkRfVG9rZW46VHJVeWJTTndibzRaVEd4ajc5aWNwUENmbnRoXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

对于dateframe的切片和获取：

例如：

`print(df["name"])`    也可以通过df.name的形式（但是不可以进行赋值）

输出：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MTUyMzI3ZTZhNGM1M2M5N2EwYTNlMzNiMTVhM2Y3OWZfeXl2ZHRLcWVJd2d5dzlLSEFXZHFZSmRLVU9kU3ZWMG9fVG9rZW46RWV1dWI5WXF4b2pPMUR4dDBoMGNjMDAzbmFlXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

```
print(df[["name","age"]])
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=OTEzZDA1MjJmMGI2NTkxOWEyNmNkZWQ0N2VmNDEzMGFfSjh2TW1GdTJub1NuTnBBTkxkM0JUcjNKWFVQRHdydEJfVG9rZW46VFJQeWJMakh3b2ZFTW94NHFYZmNzb0tpbkJnXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

直接做切片时作用的是行而不是列 

```
df[1:3]
```

用`iloc()`函数取出dateframe 返回series类型的数组

示例：

`print(df.iloc[0])`  #取出第0行

输出：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGM5MmI3YWZjMjUwMmVjODUzZmZkMzg2MzRmYzBkMmJfVkRDQ1ZNNWEwQmZRZnd3cm5vaUhaRHlkUlBiUlVDendfVG9rZW46RGxRZ2JpUG1QbzF2M254aW1rRGNYSG85blJoXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

`print(df.iloc[0，2])`   #取出第0行，2列

输出：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MTdhMmM3OTgyMmE4ZGI2MzRmN2UxNGRmOTA0YzI1M2NfMFdjSXlaZm9nd2wyMGhWcFJla1pmeXczTFNiQVJ6aEdfVG9rZW46REh0cmJjQndub0R5ZGJ4YlpCSWN3dE4wbmhCXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

同样可以配合切片进行索引：

`print(df.iloc[:,-2:])`  #取出后两列

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MmIxNjMxZmJmOWJhYjhmNWQ5OTBmMGE1YzJhOGU0ZjFfY2p0eWVVbk1sek93ZUd0VlFlSzRIVGdqa3dUeWEyVkJfVG9rZW46TUhXR2JEUjNPb1dROUN4dFlaNWNJTDIxbmNlXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

用`loc（）`函数可以进行标签的检索

`loc(,)`***写的是标签而不是数字否则会报错 还要用冒号进行分割***

例如 但是记住用：分割

```
print(df.loc[0:2,"name":"age"])
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=NjNiZDA2MTZiMDA5ZWJmMmFiZjc0NjkwYWQzOTU3OTZfRDdvVjdjZDloSXBkZnl3bXZnY2VoQk5FODJ5TTJPb2ZfVG9rZW46THlQWmJDbG45b2dlcEN4OE42RWNnaWxYbkFlXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

#### 5.排序：

##### 1.对于Series：

两个方法：

（1）`s.sort_index`（）#根据索引进行排序

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=YzJlZjUzMDZjZjFlZmYxNmQ3MjI4MDBkZWQyMzljM2FfbjdVVzJtUnB5MVFsZkd3ZHFzV2J6Y0lBeHlxUTY0ekhfVG9rZW46S0JDNmJNWVJKbzlPNWx4cjZjRWNtZGlobmtnXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

（2）`s.sort_values()`  #根据值进行排列

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=NDU4MDQ2NjkzMzlmYjRiOTY0ZjBjYTI0ZDBjY2JiOWFfcnM0bHNmdUZreHJNdlZ6dEl2QzlmQXYyVE1TNUNRem1fVG9rZW46SXRncGJyS2RIbzkxdFZ4QnQ4WWNQOVEzbkpkXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

##### 2.对于Dateframe：

通过值进行排序：

```
df.sort_values（by = "",ascending = False）
```

***#by指定排序的字段，可以传入单列，也可以是多列***

***#ascending = True（正序排列）  ascending = False（倒序排列）***

例如使用

```
print(df.sort_values(`*`by`*` = "age",ascending=True))
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=NTA3ZDRjOTIwODQ5NTQyZjU3MjhlMmM3MTgzMTQ5NDlfYUw0QnNMRE5XVm9tU2lrSkhrdWRJdmRIVHlVRW9UVXRfVG9rZW46RktFVWJ5SGlOb3hGRUV4aUNQcGN0cXNybk5UXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

*如果传入多个字段：*

*按照优先级进行排列：*

```
print(df.sort_values(`*`by`*` = ["age","height"]))
```

优先按照年龄其次按照身高

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=NDkwNzllMDk1MmNhODI5ODY2ZjI3YmZjNTQ3NGJmZmZfWlA1aFpydHdqZ1M0NWF6aUF3RDY0RnVyTTlCMjVSV1dfVG9rZW46VTZxQmJZeHZIb1Bkamx4TkFGZWN0WVI4bkxjXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

单独指定每一个的升降序：

```
print(df.sort_values(`*`by`*` = ["age","height"],`*`ascending`*`= [True , False]))
```

\#优先按照年龄正序 身高倒序排列：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MzM1ZGU5MDFkMmY1OGY2YmJlOGJlZGE2ZTcwN2U5YzFfVmx3NzBsb1FmQWZtOERFdnNJZXN0UWZTeGJCMUlrWEpfVG9rZW46Q0hqZWJ4a28wb1lWckF4NXNJdWNkR0ppbmtoXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

#### 6.唯一值，值计数，会员

##### 1.唯一值

取唯一值和numpy相同只需要：

```
s.unique（）
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MWY5ZTM5YTJlZmZjODUwNDYzYTFmZjNiNTE4YTkzZWNfMEVyc1gzMXBLckhVVklkYWV4MHMwU092d01Tc1R5VkVfVG9rZW46U0JUc2JISHlab2FLMTR4SzMyRGNqdjZNbjBkXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

则会消除所有重复的数字

##### 2.计算重复值

获取重复值出现的次数：

```
s.valuse_counts（）
```

例如获取男性和女性的人数：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=MWY2ZmJmNWYwMWRmYzEwOWM0ZGFmYzU5YWUyNjhkODBfbmVoOXpZM1d5ZTVHUWVRUjJKZkpHQ09nS3k4eHFKZGdfVG9rZW46VExkTmJJSktBb0tuNG14ZUV3dmNUTmtLblBnXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

***#它会返回一个Series类型 而且他的索引跟名字会被替换成gender内的对象名称***

假如要知道男的有几个可以把返回的Series取出来单独输出男 例如

```
Series["man"]
```

既可以输出：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTMxYTdlNjBmYWVkMTMxOGY5YjUyMzdlZjZiODU3NTdfOHV4Y3c4cGt3cTBURTlkNWYwcm03U1lla2F0aW9nUkVfVG9rZW46Q0xHUmJCUTRib3llcXR4WDNoYWNXM2UxblBjXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

如果我想找两个人的话 我可以：

```
df[(df.name == "张三") | (df.name == "王五")]
```

寻找这两个人所在的行：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=YzI3NDhhM2MyY2I0ZjQxYmMwYzgwMzMwY2FjYTc0OGVfV3p3MGpHMnVZa3NTOUkya0IxNk54dnRsMXl1V3ZacTFfVG9rZW46REY1ZWJrSmFJb2xzZHp4OFg1bmM1VW1sbkNiXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

***或者我可以使用******isin******先获取******布尔类型*** ***再去查找***

```
print(df.name.isin(["张三","王五"]))
```

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=Nzg2MjNlMmNkNjgyZjRkNjE3Yjg1NjJjMjJkMmU3YzdfZlI1Mjl4aTNjSmM2RVV2MjFObktlY0dtQ2h1V2dtZlNfVG9rZW46V2VNY2JpUUtmbzlaN3l4cjZmdmNJRkVCbk9mXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

然后再使用

```
df[df.name.isin(["张三","王五"])]
```

嵌套获取所在行：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=YTc4ZmJkNDBjNmFkMTc5YTE2OWViZTIyODkzNDEyMGFfdHNsQlpKY09ZS3hSU3RMVXp6bkk5ekpvY1BCbEJ2OTlfVG9rZW46VmVDSWJ4SnJFb2o0ekZ4dTVVdGNzZ1ZJbjRnXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

### 3数据的加载与处理

常见的读取函数：

![img](https://dcnwrbmn4oc1.feishu.cn/space/api/box/stream/download/asynccode/?code=YWFlNTdjNzc0YWNmZjY2OGQ4ZWNkMDk5NWIyMDJkNmVfQ0xRWTd2WWw1NEdZaDFRTGhaa1hNaGd2VzJWNEFuWFJfVG9rZW46T2NPbWJSUnNOb1ZIY254bGxWRGNGZXRDbnlmXzE3Mjg3MjgxODY6MTcyODczMTc4Nl9WNA)

