apple_height = list(map(int, input().split()))
taotao = int(input())
total = 0
max_reach = taotao + 30
for h in apple_height:
    if h <= max_reach:
        total += 1
print(total)
