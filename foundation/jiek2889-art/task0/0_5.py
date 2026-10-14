num = int(input())
end = int(num**0.5)
is_prime = True
for i in range(2, end + 1):
    if num % i == 0:
        is_prime = False
        break
if is_prime:
    print("yes")
else:
    print("no")
