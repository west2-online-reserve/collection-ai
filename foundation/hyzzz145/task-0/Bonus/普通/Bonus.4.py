list = [1,2,7,3,6,"asd"]
for i in list:
    if type(i) == str:
        list.remove(i)
list_len = len(list)
run = 1
while run:
    is_run = 0
    for i in range(list_len-1):
        if list[i] > list[i+1]:
            list[i],list[i+1] = list[i+1],list[i]
            is_run = 1
    if not is_run:
        run = 0
print(list)
