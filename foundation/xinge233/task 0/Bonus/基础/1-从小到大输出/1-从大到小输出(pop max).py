a,b,c=map(int,input().split())
lst=[a,b,c]
result = []
while len(lst)>0:
    result.append(lst.pop(lst.index(max(lst))))
print (*result)