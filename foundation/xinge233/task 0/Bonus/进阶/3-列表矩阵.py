lst=["1"*10 for i in range(5)]
for i in range(5):
    print("".join(lst[i]))
new_lst=[[lst[x][y] for x in range(5)] for y in range(10)]
print(new_lst)
for i in range(10):
    print("".join(new_lst[i]))
