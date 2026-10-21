def a(list):
    t={}
    for i in list:
       if i not in t:
           t[i]=1
       else:
           t[i]+=1
    return t
print(a([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]))