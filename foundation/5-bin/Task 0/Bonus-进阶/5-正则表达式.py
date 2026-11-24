import re

def main():
        pattern = r"[\da-zA-Z]{6,18}"
        s = input("请输入密码：")
        ret = re.fullmatch(pattern, s)
        while True:
            if ret is None:
                s = input("密码不符合规范，请重新输入密码：")
                ret = re.fullmatch(pattern, s)
            else:
                 print("密码符合规范，循环结束")
                 break

if __name__ == "__main__":
    main()