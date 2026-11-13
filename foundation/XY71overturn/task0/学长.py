zongrenshu=int(input("请输入总人数:"))
count=1
results=[]
for i in range(zongrenshu):
    mingzi=input("请输入姓名:")
    results.append(mingzi)
    count+=1
记录总数=int(input("请输入记录总数:"))
for i in range(记录总数):
    guanxi=input().split()
    results[int(guanxi[0])-1]="I_love_"+str(results[int(guanxi[1])-1])
print(str(results[0]))    
