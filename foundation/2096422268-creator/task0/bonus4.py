lst = eval(input())

for i in range(len(lst)-1,-1,-1):
    if not isinstance(lst[i],int):
        lst.pop(i)

lst.sort()
print(lst)