def main():
    a, b, c = map(int, input().split())
    #1
    if a > b and a > c:
        if b > c:
            print(a, b, c)
        else:
            print(a, c, b)
    elif b > a and b > c:
        if a > c:
            print(b, a, c)
        else:
            print(b, c, a)
    elif c > a and c > b:
        if a > b:
            print(c, a, b)
        else:
            print(c, b, a)
    #2
    if a < b:
        a, b = b, a
    if a < c:
        a, c = c, a
    if b < c:
        b, c = c, b
    print(a, b, c)
    #3
    l = [a, b, c]
    l.sort()
    print(*l[::-1])
if __name__ == "__main__":
    main()