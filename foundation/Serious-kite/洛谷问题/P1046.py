appletree = [105,120,125,135,140,155,160,175,180,195] #苹果树的高度
height = 110 #陶陶手所能够摘到的苹果树的高度

#计数器
count = 0

#迭代列表
for i in appletree :
    #如果苹果树的高度小于等于陶陶手所能够摘到的高度加上椅子的高度，则计数器加1
    if i <= height + 30 :
        count += 1

print(count)#5个