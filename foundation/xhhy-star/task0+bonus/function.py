def count_number(nums):
    """
    统计纯数组列表中一个数字的出现次数。

    :param nums: 纯数组列表
    :return: 数字的出现次数
    """

    result_dict = {}
    for i in nums:
        if i in result_dict:
            result_dict[i] += 1
        else:
            result_dict[i] = 1
        
    return result_dict

# 测试使用
test_nums = [1, 2, 3, 4, 5, 1, 2, 3, 1]
print(count_number(test_nums))