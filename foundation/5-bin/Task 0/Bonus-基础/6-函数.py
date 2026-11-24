def dictnum (lis) :
    dic = {}
    for i in lis :
        dic[i] = 0
    for i in lis :
        dic[i] += 1
    return dic

# lis = [1,2,3,4,5,4,5,5,6,7]
# lis = eval(input())
lis = [x for x in input().split()]
print(dictnum(lis))