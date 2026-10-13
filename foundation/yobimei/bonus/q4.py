raw = input().split()
new_list = []
for i in raw:
    if i.isdigit():
        new_list.append(int(i))
new_list.sort()
print(new_list)        