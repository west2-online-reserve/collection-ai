dict1 = {}
n = int(input())
for i in range(n):
    senpai = input()
    dict1[i+1] = senpai # key: i+1, value: senpai   i+1要不要转成str?? 不用欸
m = int(input())
for i in range(m):
    u, v = [int(j) for j in input().split()]
    dict1[u] = f'I_love_{dict1[v]}'             # u要不要转成str??不用
print(dict1[1])
#key 只要可哈希就行，无命名规则