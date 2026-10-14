apples = list(map(int, input().split()))
hand_height = int(input())
total_height = hand_height + 30

count = 0
for apple in apples:
    if apple <= total_height:
        count += 1

print(count)