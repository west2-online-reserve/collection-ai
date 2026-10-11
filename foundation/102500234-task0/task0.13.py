import re

def check_password(pwd):
    pattern = r'^[A-Za-z0-9]{6,18}$'
    if re.fullmatch(pattern, pwd):
        return True
    else:
        return False

#测试
#print(check_password("abc123"))     #True
#print(check_password("Abc12345"))  #True
#print(check_password("abc"))       #不足6位 False
#print(check_password("abc123456789012345678")) #超过18 False
#print(check_password("abc@123"))   #有特殊符号 False
#print(check_password("密码123"))    #中文 False
