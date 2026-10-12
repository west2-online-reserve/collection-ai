def main():
    students = {
    '1': '王大锤',
    '2': '李小美',
    '3': '张三丰',
    '4': '赵敏',
    '5': '周芷若',
    '6': '张无忌',
    }
    students2 = {key:value for key, value in students.items() if int(key) % 2 != 0} #deepseek
    print(students2)
if __name__ == "__main__":
    main()
