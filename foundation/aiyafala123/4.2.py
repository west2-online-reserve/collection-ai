h = list(map(int, input().split()))
h_max = int(input())
num = 0
for i in range(len(h)):
    if h[i] <= h_max + 30:
        num += 1
print(num)
