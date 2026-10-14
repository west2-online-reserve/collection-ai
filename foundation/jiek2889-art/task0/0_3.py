heights = []
while len(heights) < 10:
    need = 10 - len(heights)
    line = input()
    heights += [int(x) for x in line.split()]
max_reach = int(input())
num = 0
for i in heights[:10]:
    if i <= max_reach + 30:
        num += 1
print(num)
