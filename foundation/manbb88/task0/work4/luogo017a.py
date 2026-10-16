def oula(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    prime = []
    for i in range(2, n+1):
        if is_prime[i]:
            prime.append(i)
        for j in prime:
            if i * j > n:
                break
            is_prime[i * j] = False
            if i % j == 0:
                break
    return is_prime

is_primes = oula(100010)

def main():
    n = int(input())
    if is_primes[n]:
        print("Yes")
    else: 
        print("No")

if __name__ == "__main__":
    main()