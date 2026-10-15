apple_heights = [int(i) for i in input().split()]
tao_height = int(input())
total_height = tao_height + 30
num = 0
for i in apple_heights:
    if total_height >= i:
        num += 1
print(num)