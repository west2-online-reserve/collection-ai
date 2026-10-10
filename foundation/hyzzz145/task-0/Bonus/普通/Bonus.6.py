def count(a : list):
    dict = {}
    for  i in a:
        if i in dict.keys():
            dict[i]+=1
        else:
            dict.update({i:1})
    return dict
a = [1,2,1,4,1,5,1,6,1,76,9,1,2,5,1]
print(count(a))

