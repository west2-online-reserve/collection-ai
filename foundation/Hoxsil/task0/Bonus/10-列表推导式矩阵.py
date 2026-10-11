# emm，全是 1 有点不好观察，稍微改了一下
list1 = [[[y] for x in range(1, 6)] for y in range(1, 11)]
list2 = [[list1[x][y] for x in range(10)] for y in range(5)]
print(list1)
print(list2)