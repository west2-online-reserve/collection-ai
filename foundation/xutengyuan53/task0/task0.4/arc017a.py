import math

def main():
    n = int(input())
    
    if n == 1:
        print("NO")
        return
    
    limit = int(math.isqrt(n))
    
    is_prime = True  
    for i in range(2, limit + 1):
        if n % i == 0:
            is_prime = False
            break       
    
    if is_prime:
        print("YES")
    else:
        print("NO")

if __name__ == "__main__":
    main()