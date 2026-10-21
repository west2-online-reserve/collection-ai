a={1:'zln',2:'sjj',3:'ljj',4:'qrh'}
list1=list(a.keys())
for i in list1:
    if i%2==0:
        del a[i]
print(a)