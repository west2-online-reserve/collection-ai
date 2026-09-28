n = int(input())
#建立一个字典，存储每个人的名字
namesdict = {}
for i in range(n):
    #用i+1作为键，可以准确对应相应学长，输入的名字作为值存入字典
    namesdict[i+1] = input().strip()
m = int(input())
#根据输入的移情别恋次数，更新字典中每个人的名字
for j in range(m):
    line = input()
    p = line.split()#把输入的学号分开
    u = int(p[0])
    v = int(p[1])
    namesdict[u] = "I_love_"+namesdict[v]#更新学长的新名字
print(namesdict[1])
