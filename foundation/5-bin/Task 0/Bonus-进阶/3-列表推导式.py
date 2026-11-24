def main():
    list1 = [[1 for x in range(0, 10)] for x in range(0, 5)]
    for i in range(0,5):
        print(list1[i])

    list2 = [[list1[i][j] for i in range(0,5)] for j in range(0,10)]
    for i in range(0,10):
        print(list2[i])

if __name__ == "__main__":
    main()
