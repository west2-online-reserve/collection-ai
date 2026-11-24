#ls = [x for x in input().split()]
ls = eval(input())
num = []

for i in ls :
    #if i.isdigit() :
    if isinstance(i, int) :
        num.append(int(i))

num.sort()
# for i in num :
#     print(i, end=' ')
print(num)
