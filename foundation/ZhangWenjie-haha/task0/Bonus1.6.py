def count(num):
    cnt = {}
    for x in num:
        for c in x:
            if int(c) in cnt:
                cnt[int(c)] += 1
            else:
                cnt[int(c)] = 1
    return cnt


def main():
    arr = input().split()
    print(count(arr))


if __name__ == '__main__':
    main()
