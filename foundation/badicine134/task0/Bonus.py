# 01

list1 = [int(i) for i in input('输入三个整数：').split()]
list1.sort()
list1.reverse()
print(list1)

# 03

msg = input('输入一个字符串：')
if msg.find('ol'):
    msg = msg.replace('ol','fzu')
print(msg[::-1])

# 05
dict1 = {11:'kaka01',12:'kaka02',13:'kaka03',14:'kaka04'}
for i in list(dict1.keys()):
    if i % 2 == 0:
        del dict1[i]
print(dict1)

# 06  这题是通过ai学习的
def count(num_list):
    num_count = {}
    for i in num_list:
        num_count[i] = num_count.get(i, 0) + 1 #此处get语法由ai教学
    return num_count
print(count([1,2,3,2,1,1,2,3,1]))