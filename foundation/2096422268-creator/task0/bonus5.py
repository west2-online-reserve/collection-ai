students={
    "0001":"Huang",
    "0002":"Wang",
    "0003":"Chen",
    "0004":"Li",
    "0005":"Zheng"
}

for i in list(students.keys()):
    if int(i)%2==0:
        del students[i]

print(students)