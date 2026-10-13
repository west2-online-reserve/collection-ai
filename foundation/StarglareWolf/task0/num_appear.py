def num_times(lis):
    dic = {}
    for num in lis:
        if num not in dic.keys():
            dic[num] = 1
        else:
            dic[num] += 1
    print(dic)

lis = [1, 1, 1, 2, 3, 5, 6, 9, 8, 7, 4, 4]
num_times(lis)