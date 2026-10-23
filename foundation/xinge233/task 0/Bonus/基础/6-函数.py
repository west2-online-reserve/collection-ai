# 6. 创建一个函数，这个函数可以统计一个只有数字的列表中各个数字出现的次数，通过字典方式返回
def count_it(lst: list):
    result = {}
    for i in lst:
        result.setdefault(i, lst.count(i))
    return result


if __name__ == "__main__":
    x = [99, 23, 23, 87, 87, 87, 0, -100]
    print(count_it(x))
# from collections import Counter

# nums = [1, 1, 2, 3, 5, 5, 6]
# c = Counter(nums)将统计各个元素出现的次数，以字典的形式保存
# print(c)
# 这里是从deepseek学来的
