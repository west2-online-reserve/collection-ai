n = int(input())
name = [""]  # 空字符串用来占位，这样就能直接使用 name[1] 表示 1 号学长。
for i in range(n):
    s = input()
    name.append(s)

m=int(input())

for i in range(m):
    u,v=map(int,input().split())
    name[u]="I_love_"+name[v]

print(name[1])