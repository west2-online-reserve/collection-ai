import re
def set_password(a = None):
    if a == None:
        a = input("请输入一个长度在 6~18 之间,只能包含英文和数字的密码:")
    m = re.findall("[a-zA-Z0-9]{6,18}", str(a))
    if m:
        print ("密码设置成功")
        return a
    else:
        print ("我超威你密码给我输好了啊")

#测试
w1 = set_password(4390930)
print(w1)
