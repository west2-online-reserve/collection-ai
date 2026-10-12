def main():
    lst = ["apple", 3, "banana", 1, 7, "fzu", 5]
    i = 0
    while i < len(lst):
        if isinstance(lst[i], str):
            lst.pop(i)
        else:
            i += 1
    lst.sort()
    print(lst)
if __name__ == "__main__":
    main()