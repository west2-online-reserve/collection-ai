enter=input("请输入字符，以空格分离")
list=[]
new_list=[]
for i in enter.split():
    list.append(i)
#print(list)
for i in range(len(list)):
    try:
        new_list.append(int(list[i]))
    except ValueError:
        pass    #try是把这个代码喂给deepseek学来的
#for i in range(len(list)):
#    if (type(list[i]) == str):
#        list.remove(list[i])
print(sorted(new_list))



