a,b,c=map(int,(input().split()))
lst=[a,b,c]
result = []
x=0
zheng_lst=[i for i in lst if i>0]
fu_lst=[i for i in lst if i<=0] #将数分成正数和负数
while len(zheng_lst)>0:
    for i in zheng_lst:#先判断一波，防止一开始加完直接出错
             if i == 0:#如果正数中有某个数字被减成0了
                result.insert(0,x)#在最前面插入这个数字
                zheng_lst = [i for i in zheng_lst if i != 0 ]#去掉正数中被减成0的元素
    zheng_lst=[i-1 for i in zheng_lst]#不断减1
    x +=1#映射一波
x=0    
while len(fu_lst)>0:#负数同理
    for i in fu_lst:
            if i == 0:
                result.append(x)
                fu_lst = [i for i in fu_lst if i != 0 ]
    fu_lst=[i+1 for i in fu_lst]
    x -=1
print(*result)


    