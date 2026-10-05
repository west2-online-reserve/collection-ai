#5. 写一个正则表达式，用于验证用户密码，长度在 6~18 之间，只能包含英文和数字
import re
rule=re.compile(r"^[A-Za-z0-9]{6,18}$")
while True:
    password=input("请输入用户密码,长度在 6~18 之间，只能包含英文和数字:")
    #print(repr(password))
    #print(len(password))
    m=rule.match(password)
    if m:
        print("你过关")
        break
    else:
        print("所有人给我回来，重审一次")
        
        


                  
