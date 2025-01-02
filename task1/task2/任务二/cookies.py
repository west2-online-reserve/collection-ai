from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from lxml import etree
import json
import time
import csv
import re
import json
import datetime
import requests
import pandas as pd

s = Service("./chromedriver.exe")
browser = webdriver.Chrome(service=s)
browser.set_page_load_timeout(20)
browser.set_script_timeout(20)
try:
    browser.get("https://www.zhihu.com")
except:
    browser.execute_script("window.stop()")
time.sleep(5)
input("请扫码登陆！登陆后按enter")
cookies = browser.get_cookies()
print("cookies", cookies)
jsonCookies = json.dumps(cookies)
with open("cookies_zhihu.txt",'w') as f:
    f.write(jsonCookies)
time.sleep(5)
browser.quit()
