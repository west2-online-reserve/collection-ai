#正则表达式(有必要学一下正则表达式的基础语法)
import re

def check_password(password):
    check = r"^[A-Za-z0-9]{6,18}$"
    if re.match(check, password):
        return True
    return False

print(check_password(" __"))
print(check_password("abcdefgh"))