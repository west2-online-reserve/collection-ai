def count(list1):
    a = {}
    for x in list1:
        if x not in a:
            a[x] = 1
        else:
            a[x] += 1
    return a


list1 = [1, 4 , 4, 3, 2, 5, 1]
dict1 = count(list1)
dict1 = dict(sorted(dict1.items()))
print(dict1)