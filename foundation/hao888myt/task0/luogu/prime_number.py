## 老大洛谷这题没法提交啊
num_str: str = input()
num: int = int(num_str)
is_prime = True

for i in range(2, int(num ** 0.5) + 1):
    if num % i == 0:
        is_prime = False
        break

if is_prime:
    print("YES")
else:
    print("NO")