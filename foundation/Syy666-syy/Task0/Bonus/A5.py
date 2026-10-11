import re
pattern=r"^[A-Za-z0-9]{6,18}$"
password=input("请输入密码:")
if re.fullmatch(pattern,password):
    print("密码合法")
else:
    print("密码不合法")