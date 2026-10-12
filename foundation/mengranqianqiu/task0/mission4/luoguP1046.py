def main():
    apple_height = list(map(int, input().split()))
    tao_height = int(input())
    tao_chair_height = tao_height + 30
    count = 0
    for i in apple_height:
        if i > tao_chair_height:
            count += 1
    print(count)
if __name__ == "__main__":
    main()