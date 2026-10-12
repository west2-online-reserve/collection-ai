import math
def main():
    n = int(input())
    if n % 2 == 0:
        print("NO")
        return
    m = int(math.sqrt(n))
    for i in range(3, m + 1, 2):
        if n % i == 0:
            print("NO")
            return
    print("YES")
if __name__ == "__main__":
    main()
    