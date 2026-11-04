import re

def check_password(password):
    # 定义正则表达式
    pattern = r"^[a-zA-Z0-9]{6,18}$"
    
    # 使用 re.match 进行匹配
    if re.match(pattern, password):
        return True
    else:
        return False

pw=input()
print(check_password(pw))