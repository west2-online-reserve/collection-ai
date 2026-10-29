import json,os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
#进入文件
os.chdir(os.path.dirname(os.path.abspath(__file__)))

url = "https://www.zhihu.com/" 

#drive
drive = webdriver.Edge()
drive.get(url)
input("请在网页输入登录信息后回车")

# 保存 cookies 到本地
cookies = drive.get_cookies()
with open("cookie.json", "w", encoding="utf-8") as f:
    json.dump(cookies, f, ensure_ascii=False,indent=2)

print(cookies)
print("已保存")
drive.close()
