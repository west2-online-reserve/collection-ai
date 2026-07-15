import numpy as np

# np.random.randint(low, high=None, size=None, dtype=int)  其中high不包含本身，即左闭右开 只有low是必须的
# size用元组的表达方式

points_A = np.random.randint(low=0, high=101, size=(5, 2))
points_B = np.random.randint(low=0, high=101, size=(8, 2))

# print(points_A)   5个点,每个点2个坐标
# print(points_B)

# np.newaxis 的核心作用是给数组新增一个 “长度为 1” 的维度，不会改变数组的元素总数，
# 只是调整维度结构，最常用在触发广播、维度对齐的场。

points_A_1 = points_A[:, np.newaxis, :]
# print(points_A_1)  5组、每组1个点、每个点2个坐标

points_B_1 = points_B[np.newaxis, :, :]
# print(points_B_1)

# 这里先计算差值并广播
temp = points_A_1 - points_B_1
temp_1 = temp ** 2  # 平方
# 沿最后一维进行求和
temp_2 = np.sum(temp_1, axis=2)
# 最后进行开方
distance_matrix = np.sqrt(temp_2)
print("距离矩阵(distance_matrix):")
print(distance_matrix)

print("\n min_distance:")
print(distance_matrix.min(axis=1))

# 这个部分由豆包提醒
mask = distance_matrix < 20
# print(mask)

mask_any = mask.any(axis=0)  # 等于np.any(mask, axis=0)
# print(mask_any)

valid_indices = np.where(mask_any)[0]  # [0]取索引数组（np.where返回元组）
print(valid_indices)

# 函数形式
# np.min(a, axis=None, out=None, keepdims=False, initial=np._NoValue, where=True)
# 数组方法形式（推荐）
# a.min(axis=None, out=None, keepdims=False, initial=np._NoValue, where=True)

# axis
# 指定计算最小值的维度：
# - None（默认）：计算整个数组的最小值
# - 整数：沿指定轴计算（0 = 列，1 = 行，多维可指定更高维度）
# - 元组：沿多个轴计算
