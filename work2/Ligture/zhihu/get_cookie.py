from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import json

options = webdriver.ChromeOptions()
#设置chrome位置
options.binary_location = r"D:\learn-AI\learn-AI\task2\zhihu\chrome\chrome.exe"
#设置webdriver服务
service = Service(r"D:\learn-AI\learn-AI\task2\zhihu\chrome\chromedriver.exe")
driver = webdriver.Chrome(service=service, options=options)
driver.get("https://www.zhihu.com/signin")
input('登录后按回车')
cookies = driver.get_cookies()
with open("cookies.json",'w',encoding='utf-8') as f:
    f.write(json.dumps(cookies,ensure_ascii=False))
driver.quit()
