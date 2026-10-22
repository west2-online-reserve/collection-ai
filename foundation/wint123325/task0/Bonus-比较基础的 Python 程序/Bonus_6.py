def f(a):
    num={}
    for i in a:
        if i in num:
            num[i] += 1
        if i not in num:
            num[i] = 1
    return num        
#print(f([1,2,3,2,23,23,45,6,3,45,23]))
