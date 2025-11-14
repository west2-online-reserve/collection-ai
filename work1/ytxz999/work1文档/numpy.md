**基础**
只保存一种数据类型
```python
import numpy as np
arr=np.array([1,2,3]) #整形
arr=np.array([1.0,2,3]) #浮点型
```
数组类型转换
整数型数组与浮点型做运算也会变成浮点型
```python
import numpy as np
arr=np.array([1,2,3])
arr1=arr.astype(float)
print(arr+0.0)
```
二维数组
如[1 2 3]一维数组形状3或（3，）
[ [ 1 2 3 ] ] 一维数组形状（1,3）
 三维数组形状3（1,1,3）
 可以用ones来传入形状，
 可以用shape来查看形状
 数组维度的转换可以使用reshape
 ```python
 #创建二维数组
 arr2=np.arange(10).reshape(2,5)
 #降级为一维数组
 arr1=arr2.reshape(-1)
 #用-1让他自己计算长度
 ```
 **创建**
 ```python
 import numpy as np
 #创建一维数组--向量
 arr1 = np.array( [1,2,3] )
 #创建二维数组--行矩阵
 arr2 = np.array( [ [1,2,3] ] )
 #创建二维数组--列向量--很消耗内存
 arr3 = np.array( [ [1],[2],[3] ] )
 ```
 用arrange来创建递增数组（不包括后一个参数）
 
```python
arr1 = np.arrange(10)     #创建从0到10的递增数组
arr1 = np.arrange(10,20)  #创建从10到20的递增数组
arr1 = np.arrange(1,21,2) #创建从1到21,步长为2的递增数组
```
 创建全0数组和全1数组(浮点型)
 ```python
 arr1=np.zeros(3)
 arr2=np.ones((1,3))
 ```
 创建随机数组
 ```python
 arr1=np.random.random((5,5))   #浮点型  0到1之间
 arr1=np.random.randint(10，100，(5,5))  #10到00的整形
 arr1=np.random.normal(0，1，(5,5))   #正态分布
 ```
 用np.array来创建指定数组
 用print(arr[1,1])这种语法来访问元素，可以使用负数
 花式索引用两个中括号print(arr[ [0,2] ])返回[arr[0],arr[2]]用于同时索引多个元素
 ```python
 arr2 = np.arange(1,17).reshape(4,4)
 print(arr2[0,1],[0,1])
 #输出【1，6】也就是arr【0，0】和arr【1，1】
 ```
 普通索引                arr[x1]             arr[x1,y1]
 花式索引                arr[ [ x1,x2,x3,x4...xn ] ]             arr[ [x1,x2,x3...] ,[y1,y2,y3...]]
 **切片**
 数组的切片看python文档
 arr[ : :3]切片从头到尾，步长为3
 矩阵的切片
 ```python
 print(arr[1:3,1:-1])
 print(arr[::3,::2])      #跳跃切片
 
 ```
 提取行列
 ```python
 print(arr[2])
 print(arr[:,2])
 ```
 copy()方法可以复制数组，矩阵
 ```python
 arr1=arr2
 #arr2修改，arr1也会变化，与cpp不同
 #为了让arr1不被修改，使用copy()
 arr1=arr2.copy()
 
 ```
 向量需要使用转置需要先变成矩阵
 ```python
 arr1=[1,2,3]
 arr1=np.reshape(1,3)
 #或者
 arr1=np.reshape(1,-1)
 arr1=arr1.T
 
 
 
 
 #法2
 arr1=np.reshape(-1,1)
 ```
 **翻转**
 使用np.flipud()    这个是上下翻转
 使用np.fliplr()      这个是左右翻转
 向量只能使用前者
 **拼接**
 向量
 ```python
 arr2=np.concatenate[[arr1,arr2]]
 ```
 矩阵
  ```python
 arr2=np.concatenate[[arr1,arr2]]    #默认axis =0行拼接
 arr2=np.concatenate[[arr1,arr2],axis=1]    #列拼接
 ```
 **分割**
 ```python
  arr1,arr2 = np.split( arr ,[1,3] , axis=1 )       #列分裂
 ```
 计算
 含/会生成浮点型，//是生成整形
 **广播**
 **函数**
 abs()绝对值
 sin（）cos( ) tan( ) 三角函数
 exp( ) ,2** x 指数函数
 np.log（）/np.log（2）对数
 聚合函数
# Matplotlib

 ```python
 import matplotlib.pyplot as plt
 %matplotlib inline
 #展示高清图
 from matplotlib_inline import backend_inline
 backen_inine.set_matlplotlib_formats('svg')
 x=[1,2,3,4,5]
y=[1,8,27,64,125] 
y1=[1,2,3,4,5]
 #matlab画图
 Fig1 = plt.figure()
 plt.plot(x,y)
 #面向对象方式
 Fig2 = plt.figure()
 ax2 = plt.axes()
 ax2.plot(x,y) 
 
 #绘制子图 
 参数依次是 行 列 顺序
 plt.subplot(2.1.1),plt.plot(x,y)
 plt.subplot(2.1.1),plt.plot(x,y1)
 ```