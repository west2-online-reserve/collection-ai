import numpy as np
#arr1 = np.array([1,2,3])
#arr2 = arr1.astype(float)>>[1. 2. 3.]
#arr2 = arr1 / 2 >> [0.5 1.  1.5]

#arr1 = np.array(range(10))
#arr2 = arr1.reshape((5,-1))

#arr1 = np.arange(10)>>[0 1 2 3 4 5 6 7 8 9]
#arr1 = np.arange(1,10,0.5)>>[1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5 7.  7.5 8.  8.5 9.  9.5]

#arr1 = np.zeros(3) >>[0. 0. 0.]
#arr1 = np.zeros((2,3))#>>[[0. 0. 0.][0. 0. 0.]]
#arr1 = np.random.random(5)
#arr1 = np.random.randint(2,5,(3,5))
#arr1 = 10*np.random.random((10,5)) + 10
#arr1 = arr1.astype(int)
#arr1 = np.random.randn(2,3)
#arr1 = np.random.normal(0,1(2,3))

#arr1 = np.arange(0,100,10)

#print(arr1[[0,2,3,1]])#>>[ 0 20 30]

#arr1 = np.arange(20)
#arr2 = arr1.reshape(2,-1)

#arr2 = arr2[:5]
#print(arr2)

#arr2 = np.arange(20).reshape(4,5)
#print(arr2)

#arr1 = np.arange(10)
#arr1 = arr1.reshape(1,-1)
#arr2 = arr1.T
#print(arr2)

#arr1 = np.array(10)

#copy = arr1[ : 3].copy()
#copy[0] = 100>>[100   1   2]
#print(copy)
#print(arr1)

#arr1 = np.arange(10)
#arr2 = arr1 
#arr2[1] = 199
#print(arr1)
#print(arr2)

#翻转

#arr1 = np.arange(10)
#arr1 = arr1.reshape((2,-1))
#左右
#arr2 = np.fliplr(arr1)
#上下
#arr2 = np.flipud(arr1)
#若arr为向量，则只能用上下
#print(arr2)

#arr1 = np.array([1,2,3])
#arr2 = np.array([3,4,5])
#arr3 = np.concatenate([[arr1,arr2],[arr2,arr1]])
#print(arr3)

#arr1 = np.array([[1,2,3],[3,4,5]])
#arr2 = np.array([[12,234,31],[121,21,345]])
#行瓶贴（axis = 0）
#arr3 =np.concatenate([arr1,arr2])
#列拼贴
#arr4 = np.concatenate([arr1,arr2],axis = 1)
#print(arr4)

#分裂
#arr = np.arange(10,100,10)
##print(arr)
#arr1,arr2,arr3 = np.split(arr,[2,8])
#print(arr1,arr2,arr3)
#print(np.split(arr,[2,8]))
#print(type(arr).__name__)

#arr = np.arange(1,9).reshape([2,-1])
#arr1,arr2 = np.split(arr,[1])
#print(arr1,'\n\n',arr2)
#arr3,arr4,arr5 = np.split(arr,[1,3],axis=1)
#print(arr3,'\n',arr4,'\n',arr5)

#arr1 = np.arange(5)
#arr2 = np.arange(5)
#print(np.dot(arr1,arr2)) >> 30

#arr = np.array([-10,0,10])
#arr1 = np.abs(arr)
#theta = np.arange(3) * np.pi/2
#sin_v = np.sin(theta)
#x = np.arange(1,4)
#ex = np.exp(x)
#ln = np.log(x)

#arr = np.arange(10).reshape([2,-1])
#max = np.max(arr,axis=0)>>[5 6 7 8 9]
#max = np.max(arr,axis=1) >>[4 9]
#print(max)
#sum
#prod求积
#mean 均值
#std 标准差
#在这些函数前加 nan可以忽略缺失值


#arr = np.arange(15).reshape([3,-1])
#print(arr[arr>4])

#arr = np.random.normal(500,70,[3,7])
#print(arr)
#print(np.where(arr >600)[0],np.where(arr >600)[1])
import pandas as pd
print("hello world")
#dict_v = {"a":1,"b":2,'c':3,'d':4,"e":5}
#sr = pd.Series(dict_v)
#print(sr)
##v = [1,2,3,4,5]
#k = ['a','b','c','d','e']
#k_1 = np.array(k)
#sr = pd.Series(v,index=k_1)
#print(sr)
#print(type (sr.values))

#k = ['1号','2号','3号','4号','5号']
#v_1 = [12,35,32,12,77]
# v_2 = ['w','w','m','m','w']
# p_1 = pd.Series(v_1,index = k)
# p_2 = pd.Series(v_2,index = k)
# print(p_1)
# print(p_2)
# sr = pd.DataFrame({"年龄":p_1,"性别":p_2})
# print(sr)

# list = np.array([[12,'w'],[32,'m'],[27,'w']])
# ind = ['1号','2号','3号']
# col = ['年龄','性别']
# sr = pd.DataFrame(list,index=ind,columns=col)
# print(sr)
# print(sr.loc['1号'])
# print(sr.loc[['1号','2号']])
# print(sr.iloc[1])
# print(sr.iloc[:])
# sr.loc['2号','年龄'] = '1'
# print(sr)
# 创建畸形df
# v = [[53, 64, 72, 82], ['女', '男', '男', '女']]
# i = ['年龄', '性别']
# c = ['1号', '2号', '3号', '4号']
# df = pd.DataFrame( v, index=i, columns=c )
# df = df.T
# print(df)
# #左右翻转
# df = df[:,::-1]
# #上下翻转
# df = df[::-1,:]

# 创建 sr1 和 sr2
# v1 = [10, 20, 30, 40]
# v2 = [40, 50, 60]
# k1 = ['1号', '2号', '3号', '4号']
# k2 = ['4号', '5号', '6号']
# sr1 = pd.Series(v1, index=k1)
# sr2 = pd.Series(v2, index=k2)

#isnuill
#sr.dropna(axist=)剔除存在nan的行或者列
#fillna(0)填充nan为0或者其他 'ffill'用前面的填 'bfill'用后面的填
# sr = pd.read_csv('data.csv',index_col=0)
# print(sr)
#在sr后面用head()方法只显示前五行
#describe方法可以有很多数据特征

# 一个特征：性别
# pivot_table 用于创建透视表
# 参数1 '是否生还'：要统计的数值列（默认计算均值）
# 参数 index='性别'：按性别分组作为行索引
#df.pivot_table('是否生还', index='性别')
#默认使用mean

# 重置年龄列
# pd.cut 用于将连续数值分段（分箱）
# 参数 df['年龄']：要分段的数据列
# 参数 [0, 25, 120]：分段区间，即 (0,25] 和 (25,120]
# age = pd.cut( df['年龄'], [0,25,120] )
# age

# 重置费用列
# pd.qcut 用于按分位数将连续数值等频分段（分箱）
# 参数 df['费用']：要分段的数据列
# 参数 2：将费用自动分为两部分（即中位数切分为两段，每段数据量大致相等）
# fare = pd.qcut( df['费用'], 2 )
# fare


#matlab 方式
from matplotlib import pyplot as plt
# fig_1 = plt.figure()
# x = [1,2,3,4,5]
# y = [1,4,9,16,25]
# plt.plot(x,y)

# # plt.show()
# #oop 方式
# fig_2 = plt.figure()
# ax2 = plt.axes()
# ax2.plot(x,y)
# plt.show()

# 准备数据
# x  = [ 1, 2, 3, 4, 5 ]          # 数据的x值
# y1 = [ 1, 2, 3, 4, 5 ]          # 数据的y1值
# y2 = [ 0, 0, 4, 0, 0 ]          # 数据的y2值
# y3 = [ -1, -2, -3, -4, -5 ]     # 数据的y3值

# # MatLab方式
# Fig1 = plt.figure()                     # 创建一个新的图形窗口
# plt.subplot(3,1,1), plt.plot(x,y1)      # 将窗口分为3行1列，在第1个子图中绘制 x-y1 折线图
# plt.subplot(3,1,2), plt.plot(x,y2)      # 在第2个子图中绘制 x-y2 折线图
# plt.subplot(3,1,3), plt.plot(x,y3)      # 在第3个子图中绘制 x-y3 折线图
# plt.show()

# # 面向对象方式
# Fig2, ax2 = plt.subplots(3)     # 创建一个包含3个子图的画布，ax2 是子图数组
# ax2[0].plot(x,y1)               # 在第1个子图上绘制 x-y1 折线图
# ax2[1].plot(x,y2)               # 在第2个子图上绘制 x-y2 折线图
# ax2[2].plot(x,y3)               # 在第3个子图上绘制 x-y3 折线图
# x = np.linspace(0, 10, 50)
# y = np.sin(x)

# 线型图
# plt.plot(x, y)

# # 散点图
# plt.scatter(x, y)

# # 条形图
# plt.bar(x, y)

# # 杆图
# plt.stem(x, y)

# # 阶梯图
# plt.step(x, y)

# # 误差图
# plt.fill_between(x, y - 0.2, y + 0.2, alpha=0.3)

# # 堆叠图
# plt.stackplot(x, y, y * 0.5, y * 0.2)
normal_figure = plt.figure()
x = np.arange(-5,5,0.05)
y = np.random.normal(0,1,len(x))
plt.plot(x,y)
plt.show()