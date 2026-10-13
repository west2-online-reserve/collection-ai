#1
# li = list(map(int, input().split()))
# li.sort(reverse=True)
# print(li[0],li[1],li[2])

# a,b,c = map(int(input().split))
# if a<b:
#     a,b = b,a
# if a<c:
#     a,c = c,a
# if b<c:
#     b,c = c,b
# print(a,b,c)

# a,b,c = map(int,input().split())
# x = max(a,b,c)
# y = min(a,b,c)
# c = a+b+c-x-y
# print(x,c,y)

#2
# for i in range(1,10):
#     for j in range(1,i+1):
#         print(f'{i}x{j} = {i*j}',end=' ')
#     print()

#3
# phase = input()
# phase = phase.replace('ol','fzu')
# print(phase[::-1])

#4
# s = input().split()
# nums = []
# for i in s:
#     try:
#         nums.append(int(i))
#     except:
#         continue
# nums.sort()
# print(nums)

#5
# dic = {1:'张三',2:'李四',3:'王五'}
# to_delete = []
# for i in dic:
#     if i % 2 == 0:
#         to_delete.append(i)
# for j in to_delete:
#     del dic[j]
# print(dic)

#6
# def count(li):
#     result = {}
#     for i in li:
#         if i in result:
#             result[i] += 1
#         else:
#             result[i] = 1
#     return result
# print(count([1,2,1,3,4,4,5]))

#7






