student={}
student[1]="小一"
student[2]="小二"
student[3]="小三"
student[4]="小四"
for i in list(student.keys()):
    if i%2==0: 
        del student[i]
print(student)