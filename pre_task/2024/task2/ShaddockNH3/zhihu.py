import requests as re
import pandas as pd
import numpy as np
import time
import json

#导包
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# 禁用沙盒模式 add_argument('--no-sandbox') 兼容性问题
# 保持浏览器打开状态 add_experimental_option('detach',True)

driver=webdriver.Chrome()
driver.get('https://www.zhihu.com/topic/19554298/top-answers')
time.sleep(30)  #自己扫码

question_list_with_no_select=[]

i=2

while i<=51:
    xpath=f"/html/body/div[1]/div/main/div/div[1]/div[2]/div[4]/div/div/div/div[{i}]/div/div"
    try:
        element=driver.find_element(By.XPATH, xpath)
        question_list_with_no_select.append(element)
        print(f"qestion {i-1} has been append")

        i += 1
    except:
        print(f"Element {i-1} not found, scrolling...")
        driver.execute_script(f"window.scrollBy(0, {1000});")
        time.sleep(2)


print("50个问题已经爬完了")
extracted_data=[]

for item_element in question_list_with_no_select:
    title=''
    parent_token=''
    token=''

    data_zop = item_element.get_attribute('data-zop')

    if data_zop:
        data_zop_json=json.loads(data_zop)
        title=data_zop_json.get('title', 'No Title')

    data_extra = item_element.get_attribute('data-za-extra-module')

    if data_extra:
        data_extra_json = json.loads(data_extra)
        parent_token = data_extra_json.get('card', {}).get('content', {}).get('parent_token', 'No Parent Token')
        token = data_extra_json.get('card', {}).get('content', {}).get('token', 'No Token')

    extracted_data.append({
        'title': title,
        'parent_token': parent_token,
        'token': token
    })

print("50个问题的具体数据创建完毕")

data_sum=[]
index=0

for data in extracted_data:
    index+=1

    print(f"这是第{index}个问题的答案，创建url并且跳转")

    answer_list = []

    try:
        url = 'https://www.zhihu.com/question/' + str(data['parent_token']) + '/answer/' + str(data['token'])
        driver.get(url)
        time.sleep(10)

        print("跳转完毕，开始点击查看全部")

        div_element = driver.find_element(By.XPATH, "//div[contains(@class, 'Card') and contains(@class, 'ViewAll')]")

        view_all_button = div_element.find_element(By.XPATH,
                                                   ".//a[contains(text(), '查看全部') and contains(text(), '个回答')]")

        view_all_button.click()
        time.sleep(5)

        # 获取问题的答案，20个

        print("开始获取20个数据")

        i = 2
        j = 0
        k = 0
        e = 0

        while i <= 21:
            if e == 2:
                break

            xpath = f'/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div/div/div/div/div[2]/div/div[{i + k}]/div/div/div[2]/span[1]/div/div/span'
            try:
                element = driver.find_element(By.XPATH, xpath)
                paragraphs = element.find_elements(By.TAG_NAME, 'p')

                answer_text = ""
                for p in paragraphs:
                    answer_text += p.text + "\n"

                answer_list.append(answer_text)

                print(f"answer {i - 1} has been append")
                i += 1
                j = 0
                e = 0

            except:
                j += 1
                if j == 5:
                    e += 1
                    j = 0
                    k += 1
                    print("有广告,跳过并重新检索")
                    continue

                print(f"Element {i - 1} not found, scrolling...")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

                driver.execute_script("window.scrollBy(0, -1000);")
                time.sleep(1)

                driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(1)

                time.sleep(5)
    except:
        print("网络链接有误")


    full_data = {
            'index': index,
            'question': data['title'],
            'answer_list': answer_list,
        }

    data_sum.append(full_data)

print("数据收集完毕")

df = pd.DataFrame(data_sum)
answer_df = df['answer_list'].apply(pd.Series)
answer_df.columns = [f'answer_{i+1}' for i in range(answer_df.shape[1])]
df_final = pd.concat([df.drop('answer_list', axis=1), answer_df], axis=1)
df_final.to_csv('answer.csv', index=False,encoding='utf-8-sig')
