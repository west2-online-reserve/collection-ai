from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
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
#创建一个 ChromeOptions 对象，这个对象用来配置浏览器启动时的各类参数和选项
options = webdriver.ChromeOptions()

browser = webdriver.Chrome(service=s,options=options)

url = ['https://jwch.fzu.edu.cn/info/1039/13808.htm',]
for i in range(len(url)):

    browser.get(url[i])
    time.sleep(0.5)

    for i in range(100):
        ort = browser.find_elements(By.XPATH,'//p[@class="w-main-dh-text"]/a')[2]
        browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
        text = browser.page_source
        tree = etree.HTML(text)
        times = browser.find_elements(By.XPATH,'//ul[@style="list-style-type:none;"]//span')
        list_annex = tree.xpath("//ul[@style='list-style-type:none;']/li")
        for i,annex in enumerate(list_annex):
            annex_all=[]
            annex_content=annex.xpath("./a/text()")[0]
            annex_link='https://jwch.fzu.edu.cn'+annex.xpath("./a/@href")[0]
            download = annex.xpath("./span[@id = 'nattach15643339']/text()")
            annex_all.append(annex_content)
            #print(f"{i},{annex_content}")
            annex_all.append(annex_link)
            annex_all.append(times[i].text)
            annex_all.append(ort.text)
            with open("./try.csv", "a", encoding="utf-8-sig") as f:
                write = csv.writer(f)
                write.writerow(annex_all)
            time.sleep(0.5)
        # all.append(annex_all)
        #下一页
        btn  = browser.find_element(By.XPATH,'//span[contains(text(),"下一篇")]')
        btn.click()
        time.sleep(0.5)
        browser.switch_to.window(browser.window_handles[-1])

browser.quit()

