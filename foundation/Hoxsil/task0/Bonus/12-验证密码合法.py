import re
password = input()
if re.match(r'\w{6,18}', password):
    print("你的密码合法")
else:
    print("你的密码不合法")