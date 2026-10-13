def main():
    apples = list(map(int, input().split()))
    max_height = int(input())
    end_height = max_height + 30
    count = 0
    for apple in apples:
        if apple <= end_height:
            count += 1
    print(count)

if __name__ == "__main__":
    main()