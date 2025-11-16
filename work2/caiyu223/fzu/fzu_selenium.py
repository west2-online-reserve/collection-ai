from selenium import webdriver
from selenium.webdriver.edge.service import Service as edgeservice
from selenium.webdriver.support.ui import WebDriverWait  #等待元素
from selenium.webdriver.common.by import By  #路径
import time
import csv
import pandas as pd
import re

def text_clean(text):
    result = re.sub(r'\s+', '',text )
    result = re.sub(r'[a-zA-Z]', '', result)
    result = re.sub(r'''[^\u4e00-\u9fa50-9，。！？；："'‘'“”（）《》【】]''','',result)
    result = re.sub(r'[0-9]{5,}','',result)
    result = re.findall(r'正文.*加入收藏',result)
    result = list_to_str(result)
    patterns_to_remove = [
        '正文',
        '加入收藏', 
        '字体：大中小分享到：',
        '发布时间：'
        ]
    for pattern in patterns_to_remove:
        result = re.sub(pattern, '', result)
    return result


#将列表中字符串合成一整串字符串
def list_to_str(list):
    strs = ''
    for str in list:
        strs += str
    return strs

'''
df = pd.DataFrame(columns=['title','time','department','file_dowload_situation','content'])
df.to_csv("data_selenium.csv",index=False)
print('csv初始化')
'''
service = edgeservice(r"C:\Program Files (x86)\Microsoft\Edge\Application\edgedriver_win64 (1)\msedgedriver.exe")
driver = webdriver.Edge(service=service)

page = 172
f_page = 179
get_num = 1
content = []
while page <= f_page:
    time.sleep(1)
    driver.get(url = f'https://jwch.fzu.edu.cn/jxtz/{page}.htm')
    message_window = driver.current_window_handle
    hrefs = driver.find_elements(By.XPATH,'//ul[@class="list-gl"]/*/a')
    print(f'第{page}页')
    page += 1
    for href in hrefs:
        
        try:
            href.click()
            windows = driver.window_handles
            driver.switch_to.window(windows[-1])
            time.sleep(2)
            text_content = driver.find_element(By.XPATH,'//div[@id="vsb_content"]').text
            text_content = re.sub(r'\n+', '',text_content)
            title = driver.find_element(By.XPATH,'/html/body/div[1]/div[2]/div[2]/form/div/div[1]/div/div[1]/h4').text
            post_time = driver.find_element(By.XPATH,'//span[@class="xl_sj_icon"]').text
            department = driver.find_element(By.XPATH,'/html/body/div[1]/div[2]/div[1]/p/a[3]').text
            file = [i.text for i in driver.find_elements(By.XPATH,'//ul[@style="list-style-type:none;"]//li')]

            text = {
                'title':title,
                'time':post_time,
                'department':department,
                'file_dowload_situation':list_to_str(file),
                'content':text_content
                    }
            
            df1 = pd.read_csv("data_selenium.csv")
            df1.loc[len(df1)]=text
            
            df1.to_csv("data_selenium.csv",index=False)
            get_num += 1
            print(f'{get_num}')
            
        except Exception as e:
            print(e)
        driver.close()
        driver.switch_to.window(message_window)
        time.sleep(0.5)
            


time.sleep(50)
