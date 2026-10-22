Name={
    1:"王大",
    2:"王二",
    3:"张三",
    4:"李四",
}
l=list(Name.keys())
for i in l:
    if i%2==0:
        del Name[i]
print(Name)