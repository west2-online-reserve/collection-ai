import re

def verify_password(password):

    pattern = r'^[A-Za-z0-9@#$%^&+=]{6,18}$'
    if re.match(pattern, password):
        return True
    else:
        return False
    
# 测试使用
print(verify_password("abc123"))  # 应该会输出true这个bool值喵
print(verify_password("abc"))     # 应该会输出false这个bool值喵
print(verify_password("abc123!*&$"))  # 应该会输出false这个bool值喵