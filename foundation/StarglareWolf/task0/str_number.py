def str_num(lst):
    nums = [x for x in lst if isinstance(x, int)]
    nums.sort()
    print(nums)

lst = [3, 'hello', 1, 2, "world"]
str_num(lst)