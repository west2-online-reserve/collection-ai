import re

passward = input('请输入密码：')
a = re.fullmatch(r'^[0-9a-zA-Z]{6,18}$',passward)
if a:
    print("True")
else:
    print("False")