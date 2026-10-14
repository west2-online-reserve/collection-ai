lst = eval(input())
result=sorted(filter(lambda x:isinstance(x,int),lst))
print(result)