student={
    "102501001":"zhangsan",
    "102501002":"zhaosi",
    "102301003":"wangwu",
    "102301004":"zhaoliu",
}
for s in list(student.keys()):
    if int(s[-1])%2==0:
        del student[s]
print(student)