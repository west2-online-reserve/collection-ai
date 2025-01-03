from selenium import webdriver
import os
from json import dump, load
import time
import re
import os
import csv
from selenium.webdriver.common.by import By
from lxml import etree
import requests
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys



f=open(r'C:\Users\宋志坤\Desktop\新建文件夹\学习\新建文件夹\zhihu.csv','w',encoding='utf-8-sig')#
writer=csv.writer(f)
t=0
#加上反爬伪装 能够爬的更多一些
options=Options()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_argument('--disable-gpu')
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
driver = webdriver.Chrome(options=options)
actions=ActionChains(driver)
# if os.path.exists("cookies"):
#     cookies=load(open("cookies"))
#     for cookie in cookies:
#         driver.add_cookies(cookie)
# else:
#     while driver.current_url != "https://www.zhihu.com/":
#             time.sleep(1)
#     dump(
#             [

#                 {"name":cookie["name"],"value":cookie["value"]}
#                 for cookie in driver.get_cookies()
#             ],
#             open("cookies","w"),

#     )

# 加载指定的页面
driver.get('https://www.zhihu.com/')

time.sleep(20)
# 这里手动登录

driver.get('https://www.zhihu.com/topic/19554298/top-answers')
driver.maximize_window()
time.sleep(5)
#print(page)
# question1=page.xpath('//div/a')
# print(question1)
#driver.find_element(By.XPATH,'//button')
for _ in range(1):
    #js=f"window.scrollTo(0, {x*500});"
    js="window.scrollTo(0,document.body.scrollHeight)"
    driver.execute_script(js)
    time.sleep(2)
time.sleep(5)
#拉到底 获得尽量多的问题
question1=driver.find_elements(By.XPATH, "//h2/div/a")
url_list=[]
title_list=[]
Answer=[]
question_content=[]
#问题的爬取
for question in question1:
    url_list.append(question.get_attribute("href"))
    title_list.append(question.text)
# 爬取答案
writer.writerow(title_list)
for url in url_list:
    #爬多了会寄 设置少一点...
    # t+=1
    # if(t==14):  
    #     break
    driver.get(f"{url}")
    time.sleep(5)
    
    #是否有展开的按钮
    try:
         stretch=driver.find_element(By.XPATH,'//div[@class="QuestionRichText QuestionRichText--expandable QuestionRichText--collapsed"]/div/button').click()
         question_content.append(driver.find_element(By.XPATH,'//div[@class="QuestionRichText QuestionRichText--expandable"]').text)
      
    except:
        try:
            question_content.append(driver.find_element(By.XPATH,'//div/span[@class="RichText ztext css-ob6uua"]').text)
        except:
            question_content.append("没有内容")
  
    time.sleep(5)
    try:
        driver.find_element(By.XPATH,'//div[@class="Card ViewAll"]/a').click()
        time.sleep(1)
    except:
        pass

    for i in range(20):
        #按下上箭头 模拟用户行为 不然刷新不出来
        js_real=f"window.scrollTo(0, {i*5000});"
        driver.execute_script(js_real)
        actions.send_keys(Keys.PAGE_UP).perform()
        actions.send_keys(Keys.PAGE_DOWN).perform()
        time.sleep(1)
    #找到问题答案
    answers=driver.find_elements(By.XPATH,'//div[@class="ContentItem AnswerItem"]')
    answer_text=[element.text for element in answers]
    Answer.append(answer_text)
    print(Answer)

        

    



print('*'*100)





#写入csv

driver.quit()
writer.writerow(question_content)
for answer in Answer:
    writer.writerow(answer)
# for question in title_list:
#     print(question)
#     f.write(question+',')
# f.write('\n')
# f.write("问题内容"+',')
# for q_c in question_content:
#      if q_c==1:
#          f.write("None")
#      else:
#         f.write(q_c,+ ",")
# f.write("\n")
# f.write("回答"+",")
# for answers in Answer:
#     for answer in answers:
#         print(answer)
#         f.write(answer)
#     f.write(',')

f.close()


