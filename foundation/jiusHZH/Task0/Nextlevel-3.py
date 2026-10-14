a=[[1]*10 for i in range(5)]
print(a)
a=[[1]*len(a) for i in range(len(a[0]))]
print(a)