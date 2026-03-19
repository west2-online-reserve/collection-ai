from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver # type hint
from selenium.webdriver.common.by import By

# 配置浏览器和驱动路径
from selenium.webdriver.chrome.service import Service
service = Service(executable_path=r"D:\FOREFOX\Softs\chromedriver-win64\chromedriver.exe")

# 获取
driver:WebDriver = webdriver.Chrome(service=service)

driver.get("https://www.zhihu.com")
input("请登录")

import json
cookies = driver.get_cookies()

with open(r"west2_online\web_crawler\selenium\cookies.json",'w') as f:
    json.dump(cookies,f)
    print('OVER')

driver.quit()