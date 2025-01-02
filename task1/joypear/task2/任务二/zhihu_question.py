from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from lxml import etree
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
import csv
import re
import json
import datetime
import requests
import pandas as pd

s = Service("./chromedriver.exe")
options = webdriver.ChromeOptions()

options.add_argument("--disable-extensions")# 禁止用扩展
options.add_experimental_option('useAutomationExtension', False)
options.add_experimental_option("excludeSwitches", ["enable-automation"])

browser = webdriver.Chrome(service=s,
                            options=options)
browser.delete_all_cookies()  # 清除所有cookies#可以避免之前的会话干扰测试结果
browser.set_page_load_timeout(50)#网页加载
browser.set_script_timeout(50)#脚本执行



try:
    browser.get("https://www.zhihu.com/topic/19554298/top-answers")
except:
    browser.execute_script("window.stop()")
#隐式等待
browser.implicitly_wait(5)
time.sleep(1.5)


try:

    # 先前保存过的cookie
    f1 = open("cookies_zhihu.txt")
    cookies = f1.read()
    cookies = json.loads(cookies)#json转化为python
    # Cookie 添加到浏览器会话未登录的页面中，模拟用户的登录状态
    for co in cookies:
        browser.add_cookie(co)
except:
    browser.find_element("xpath",
                        "//button[@class='Button Modal-closeButton Button--plain']").click()
browser.refresh()###很重要，添加cookie之后必须刷新重启。
time.sleep(2) 


#写入每个列标题

header = ['问题名称','问题内容']
for i in range(20):
    header.append(f"NO.{i+1}回复")
with open("./zhihuplus.csv", "w", encoding="utf-8-sig") as f:
    write = csv.writer(f)
    write.writerow(header)

body = browser.find_element(By.TAG_NAME, 'body')
# 控制滚动次数
for i in range(20):
    btns = browser.find_elements(By.CSS_SELECTOR,"[data-za-detail-view-element_name = 'Title']")
    if len(btns)>120:
        break
    browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    body.send_keys(Keys.PAGE_DOWN)
    body.send_keys(Keys.PAGE_UP)  # 按下 Page up 键 # 反正是试出来的，不然就卡着不动     
    body.send_keys(Keys.PAGE_DOWN)  # 按下 Page Down 键
    time.sleep(0.6)  # 等待页面加载

# 很玄学，不知道这玩意为啥可以起作用
browser.execute_script('window.scrollTo(0, 0);')
time.sleep(0.5)

#取前50道问题（包括重复的）
btns = browser.find_elements(By.CSS_SELECTOR,"[data-za-detail-view-element_name = 'Title']")[:120]
question = []
for btn in btns:
    if len(question)>50:
        break
    if btn.text in question:
        continue
    question.append(btn.text)
    dic = {}
    #WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, "//h1[@class='QuestionHeader-title']")))
    dic["问题名称"] = btn.text
    #避免被覆盖，挪到该btn处
    browser.execute_script("arguments[0].scrollIntoView({ behavior: 'smooth', block: 'start' });", btn)
    btn.click()
    time.sleep(1.5)
    # 切换到新的页面
    browser.switch_to.window(browser.window_handles[-1])

    #再切换到包含全部问题的页面，不然只有3个回复
    btn = browser.find_elements(By.XPATH,"//a[contains(text(),'查看全部')]")
    if btn:
        btn[0].click()
    else:
        del question[-1]
        browser.close()
        browser.switch_to.window(browser.window_handles[0])
        continue
    time.sleep(1)
    browser.switch_to.window(browser.window_handles[-1])
    # 一加载就出问题 browser.refresh()
    time.sleep(1.5)

    # 获取新的页面高度
    body = browser.find_element(By.TAG_NAME, 'body')
    browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    for _ in range(9):  # 控制滚动次数
        last_height = browser.execute_script("return document.body.scrollHeight")
        browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
        browser.execute_script("window.scrollBy(0, 1000);")
        time.sleep(1)  # 等待页面加载        
        if last_height == browser.execute_script("return document.body.scrollHeight"):
            body = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            browser.execute_script("window.scrollBy(0, -2000);")
            time.sleep(1)
            browser.execute_script("window.scrollBy(0, 3000);")
            time.sleep(1)


    # 问题内容--点击‘显示全部’
    browser.execute_script('window.scrollTo(0, 0)')
    time.sleep(1)
    btas = browser.find_elements(By.XPATH,'//button[@class = "Button QuestionRichText-more FEfUrdfMIKpQDJDqkjte Button--plain fEPKGkUK5jyc4fUuT0QP"]')
    
    # 判断按钮是否存在
    if btas:
        btas[0].click()  # 如果元素存在，点击第一个元素
        time.sleep(0.5)

    text = browser.page_source
    tree = etree.HTML(text)

    #防止因为空问题内容导致数组越界
    cont = tree.xpath("//span[@class= 'RichText ztext css-ob6uua']//text()")
    if cont:
        dic["问题内容"] = '\n'.join(cont)
    else :
        dic["问题内容"] = 'None'
    all_reply = tree.xpath("//span[@class = 'RichText ztext CopyrightRichText-richText css-ob6uua']")[:20]
    print(len(all_reply))
    for i,reply in enumerate(all_reply):
        list_para = reply.xpath(".//text()")
        # 从列表全部整合到一个string,用换行符分隔
        reply = '\n'.join(list_para)
        dic[f"NO.{i+1}回复"] = reply

    with open("./zhihuplus.csv", "a+", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, dic.keys())#字典的写入
        writer.writerow(dic)
    # 关闭该标签页后要跳转至原先标签页，不然会找不到报错
    browser.close()
    browser.switch_to.window(browser.window_handles[0])

time.sleep(3)
browser.quit()




