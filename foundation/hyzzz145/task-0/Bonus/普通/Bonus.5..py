dict = {1:"a",2:"b",3:"c",4:"d",5:"e"}
for i in list(dict.keys()):
    if i % 2 == 0:
        del dict[i]
print(dict)
