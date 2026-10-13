n = int(input("统计学生数量:"))
dic = {}
for i in range(n):
    stu_num = input()
    name = input()
    dic[stu_num] = name

for key in dic.keys():
    if int(key[-1]) %2 != 0:
        print(f"{key}:{dic[key]}")