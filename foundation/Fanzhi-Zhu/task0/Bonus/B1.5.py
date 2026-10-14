n=int(input())
students={}
for i in range(n):
    id,name=input().split()
    students[id]=name
delete_id=[]
for id in students:
    if int(id[-1])%2==0:
        delete_id.append(id)
for id in delete_id:
    del students[id]
print(students)