my_list = ["apple", 12, "banana", 8, "cherry",66]

int_list = []

for item in my_list:
    if isinstance(item, int):
        int_list.append(item)

int_list.sort()

print("处理后的列表：", int_list)