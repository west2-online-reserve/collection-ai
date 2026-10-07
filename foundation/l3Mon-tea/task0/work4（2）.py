# G = 0
# i = 0
# h2 = int(input()) + 30
# while i <10:
#     h1=int(input())
#     if h1 <= h2:
#         G += 1
#     i += 1
# print(f"{G}")
# -----------------------------------------------------------------------
#以上为做法1
height = []
num = 0
for i in range(10):
    a = int(input())
    height.append(a)
b = int(input())
for g in height:
    if b + 30 >= g:
        num += 1
print(f"{num}")