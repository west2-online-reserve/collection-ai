apple_height = list(map(int, input().split()))
taotao_height = int(input())
max_reach = taotao_height + 30
count = 0
for height in apple_height:
    if height <= max_reach:
        count += 1
print(count)
