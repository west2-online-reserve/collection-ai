def count(lst):
    d={}
    for x in lst:
        d[x]=d.get(x,0)+1
    return d

lst=list(map(int,input().split()))

print(count(lst))