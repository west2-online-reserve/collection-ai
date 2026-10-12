def main():
    n = int(input())
    a = []
    a.append(0)
    for i in range(n):
        a.append(input())
    m = int(input())
    for i in range(m):
        x, y = map(int, input().split())
        a[x] = "I_love_" + a[y]
    print(a[1])
if __name__ == "__main__":
    main()