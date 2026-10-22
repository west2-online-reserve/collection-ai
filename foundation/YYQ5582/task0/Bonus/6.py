def num(l):
    count={}
    for i in l:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    return count