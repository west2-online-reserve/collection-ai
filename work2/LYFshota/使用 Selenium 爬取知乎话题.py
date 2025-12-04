from selenium import webdriver 
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
import time
import json
import os
import pathlib
import csv

if __name__ == "__main__":
    options = Options()
    options.add_argument('--disable-blink-features=AutomationControlled')#隐藏自动化控制标头
    options.add_argument('--ignore-certificate-errors')#忽略证书错误
    options.add_argument('--ignore-ssl-errors')#忽略SSL错误
    options.add_experimental_option('excludeSwitches', ['enable-automation'])#排除启用自动化开关
    perfs={
        'download.default_directory': str(pathlib.Path(__file__).parent / "知乎话题数据"),
        "profile.default_content_settings_values.automatic_downloads":True
    }
    options.add_experimental_option('prefs',perfs)
    browser=webdriver.ChromiumEdge(options=options)

    url='https://www.zhihu.com/'
    browser.get(url)
    #获取cookies
    if 'zhihu_cookies.json' not in os.listdir(pathlib.Path(__file__).parent / "知乎话题数据"):
        print("请在30秒内登录知乎，Cookies 文件会自动保存！")
        time.sleep(30)
        with open(pathlib.Path(__file__).parent / "知乎话题数据" / 'zhihu_cookies.json','w',encoding='utf-8') as f:
            f.write(json.dumps(browser.get_cookies(),ensure_ascii=False,indent=4))
        browser.refresh()
    else:
        browser.delete_all_cookies()
        with open(pathlib.Path(__file__).parent / "知乎话题数据" / 'zhihu_cookies.json','r',encoding='utf-8') as f:
            cookies_list=json.load(f)
            for cookie in cookies_list:
                browser.add_cookie(cookie)
        browser.refresh()

    
    finally_question_num=21
    finally_question_xpath=f'//*[@id="TopstoryContent"]/div/div/div[{finally_question_num}]/div/div/div/div/h2/div/a'
    max_scroll=15#最多滚动次数
    for _ in range(max_scroll):
        try:
            browser.find_element(By.XPATH,finally_question_xpath)
            print("已加载所有问题，停止滚动")
            break
        except NoSuchElementException:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
    #等待所有问题加载完成
    wait=WebDriverWait(browser,15)
    wait.until(EC.presence_of_all_elements_located((By.XPATH,finally_question_xpath)))

    question_records=[]
    question_num=1
    while len(question_records)<20:
        try:
            browser.find_element(By.XPATH,f'//*[@id="TopstoryContent"]/div/div/div[{question_num}]/div/div/div/div/h2/div/a')
            #过滤广告和专栏文章
            if "Pc-feedAd-new-title" in browser.find_element(By.XPATH,f'//*[@id="TopstoryContent"]/div/div/div[{question_num}]/div/div/div/div/h2/div/a').get_attribute('class'):
                question_num += 1
                continue
            if "zhuanlan.zhihu.com" in browser.find_element(By.XPATH,f'//*[@id="TopstoryContent"]/div/div/div[{question_num}]/div/div/div/div/h2/div/a').get_attribute('href'):
                question_num += 1
                continue

        except NoSuchElementException:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            question_num += 1
            time.sleep(2)
        question_element=browser.find_element(By.XPATH,f'//*[@id="TopstoryContent"]/div/div/div[{question_num}]/div/div/div/div/h2/div/a')
        question_num += 1
        question_title=question_element.text
        question_url=question_element.get_attribute('href')
        if question_url and '/answer' in question_url:
            question_url = question_url.split('/answer', 1)[0]
        question_records.append({'问题标题':question_title,'问题链接':question_url})
        
        print(f"已获取第 {len(question_records)} 个问题：")
        print(f"问题标题：{question_title}")
        print(f"问题链接：{question_url}")

    #新建保存问题CSV文件
    csv_path=pathlib.Path(__file__).parent /"知乎话题数据" /"知乎话题问题数据.csv"
    with csv_path.open("w",encoding ="utf-8",newline='') as csvfile:
        write_data=csv.DictWriter(csvfile,fieldnames=['问题标题','问题链接','问题内容'] + [f'回答{i}' for i in range(1, 11)])
        write_data.writeheader()
        write_data.writerows(question_records)


    #打开保存问题详情的CSV文件
    with csv_path.open("r",encoding ="utf-8") as csvfile: 
        question_data=list(csv.DictReader(csvfile))
    

    question_records=[]
    for row in question_data:
        link=row.get('问题链接','').strip() if isinstance(row, dict) else ''
        title=row.get('问题标题','').strip() if isinstance(row, dict) else ''
        if not link.startswith('http'):
            print(f"跳过无效链接：{link}")
            continue
        question_records.append({'问题标题':title,'问题链接':link})

    
    #获取每个问题的详细内容
    for record in question_records:
        que_url=record['问题链接']
        browser.get(que_url)
        time.sleep(2)

        #判断问题内容是否折叠，如果是则尝试展开
        try:
            content_wrapper = browser.find_element(By.XPATH,'/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div')
            wrapper_class = content_wrapper.get_attribute('class') or ''
            if "QuestionRichText--collapsed" in wrapper_class:
                expand_btn = browser.find_element(By.XPATH,'/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/button')
                expand_btn.click()
                time.sleep(1)
        except NoSuchElementException:
            pass
        time.sleep(1)
        #获取问题内容
        try:
            question_content_xpath='/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/div/span/span[1]'
            question_content_list=browser.find_elements(By.XPATH,question_content_xpath)
            for content in question_content_list:
                question_content=content.text
                #把问题内容补充到CSV文件中
                record['问题内容'] = question_content
                print(f"问题内容：{question_content}")
        except NoSuchElementException:
            pass

        #获取每个问题下的回答
        for answer_num in range(1,11):
            try:
                #每三次滚动一次页面
                if answer_num % 3 == 1:
                    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                answer_xpath=f'/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[{answer_num}]/div/div/div[2]/span[1]/div/div/span/span[1]'
                answer_content_list=browser.find_elements(By.XPATH,answer_xpath)
                for content in answer_content_list:
                    answer_content=content.text
                    #把每个回答的内容补充到CSV文件中
                    record[f'回答{answer_num}'] = answer_content
                    time.sleep(0.5)
                    print(f"第 {answer_num} 个回答内容：{answer_content}")
            except NoSuchElementException:
                print(f"问题下无第 {answer_num} 个回答，跳过")
                continue
        #保存包含回答的CSV文件
        csv_path=pathlib.Path(__file__).parent /"知乎话题数据" /"知乎话题问题数据.csv"
        with csv_path.open("w",encoding ="utf-8",newline='') as csvfile:
            write_data=csv.DictWriter(csvfile,fieldnames=['问题标题','问题链接','问题内容'] + [f'回答{i}' for i in range(1, 11)])
            write_data.writeheader()
            write_data.writerows(question_records)
        time.sleep(1)
    

