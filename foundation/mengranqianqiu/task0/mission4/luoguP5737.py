def main():
    a, b = map(int, input().split())
    c = []
    #deepseek:c = [i for i in range(a, b + 1)
    #              if (i % 4 == 0 and i % 100 != 0) or (i % 400 == 0)]
    for i in range(a, b + 1):
        if i % 4 == 0 and i % 100 != 0 or i % 400 == 0:
            c.append(i)
    print(len(c))
    print(*c)
if __name__ == "__main__":
    main()