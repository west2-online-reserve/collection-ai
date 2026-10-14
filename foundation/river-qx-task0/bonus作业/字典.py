student = {
    "2023001": "张三",
    "2023002": "李四",
    "2023003": "王五",
    "2023004": "赵六"
}

new_student = {}

for id in student:
    if int(id[-1]) % 2 != 0:
        new_student[id] = student[id]

print(new_student)