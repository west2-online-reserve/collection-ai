n = int(input())
def is_prime(num):
    if num < 2:
        return False
    i=2
    while i*i <= num:
        if num % i ==0:
            return False
        i +=1
    return True
if is_prime(n):
    print("Yes")
else:
    print("No")