num = 0
n = int(input())
for i in range(2, int(n ** 0.5) + 1): # 这里我一开始写（2，n），AI让我优化成根号n，~~AI好聪明~~
    if n % i == 0:
        num += 1
        break
if num == 0:
    print('YES')
else:
    print('NO')