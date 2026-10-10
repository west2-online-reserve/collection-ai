tree_high = input().split()
tao_high = int(input())
chair_high = 30
idx = 0
for i in tree_high:
    if int(i) <= tao_high + chair_high:
        idx += 1
print(idx)
    