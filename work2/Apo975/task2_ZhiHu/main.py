from lxml import etree
from selenium import  webdriver
from selenium.webdriver.edge.service import Service # 这里使用的是Edge浏览器
from time import sleep
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import csv
import random

DRIVER_PATH=r"D:\Driver\13.edgedriver_win64\msedgedriver.exe" # 我的驱动路径
CRAWLER_NUMBERS=20
CRAWLER_ANSWERS_NUMBERS=10
url="https://www.zhihu.com/topic/19554298/top-answers" # 知乎编程话题页面
#由于作业给的网址通向话题的 “精华 ”页面，而此页面中有不少重复问题，故爬取的内容可能有重复，后面需要进行去重处理

data_list=[] #存储爬到的所有数据

def removal(question_list):
    # 对获得的问题进行去重
    question_url_list = []  # 存储去重过的问题 url
    for question in question_list:
        temp_url = question.xpath('@href')[0]
        temp_url = temp_url.split('/answer')[0]  # 除去指向特定答案部分的 url，直接保留问题的 url
        flag = 1
        for question_url in question_url_list:
            if temp_url == question_url:
                flag = 0
                break
        if flag:
            question_url_list.append(temp_url)
        if len(question_url_list) == CRAWLER_NUMBERS:
            break
    return question_url_list

def slide():
    # 控制页面下滑,包含随机操作，反反爬
    scroll_percent = random.uniform(0.4, 0.9)
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    sleep(random.uniform(0.5, 1))
    driver.execute_script(f"window.scrollTo(0, document.documentElement.scrollHeight * {scroll_percent});")
    sleep(random.uniform(0.5, 1.5))


edge_options = Options() # Edge的配置实例

#反反爬措施
# （不知道是IP被知乎标记了还是什么原因，一开始测试的时候不需要反反爬就可以爬，后面一直跳验证码（））
edge_options.add_argument("--disable-blink-features=AutomationControlled") #要禁用 Edge的 AutomationControlled特征
edge_options.add_argument(#我的 UserAgent
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
    )

#（下面四行反反爬代码由 ai提供）
# 移除 webdriver 标志
edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
edge_options.add_experimental_option("useAutomationExtension", False)
prefs = {
    "profile.managed_default_content_settings.images": 2,  # 禁用图片
    "profile.managed_default_content_settings.javascript": 1,  # 保留 JS
}
edge_options.add_experimental_option("prefs", prefs)


driver=webdriver.Edge(service=Service(DRIVER_PATH), options=edge_options)
driver.get(url)
sleep(10) # 扫码（）
for i in range(CRAWLER_NUMBERS//10+1):
    slide()

# 获得各个问题的url
page_text=driver.page_source
tree=etree.HTML(page_text)
question_list=tree.xpath('//*[@id="TopicMain"]/div[4]/div/div/div/div/div/div/h2/div/a')

# 对获得的问题进行去重
question_url_list=removal(question_list)

print(f"共收集到{len(question_url_list)}个待爬取问题")
# 对每个问题进行处理：
num=0
for question_url in question_url_list:
    num+=1
    print(f"正在爬取第{num}个问题")
    driver.get('https://'+question_url)
    sleep(2)

    try:#尝试寻找展开按钮
        btn=driver.find_element(By.XPATH,
                                '//div[contains(@class, "QuestionRichText")]/div/button[@type="button" and contains(text(), "显示全部")]'
                                )
        btn.click()
        print("click")
    except:
        print("not click")
        pass

    #下滑刷新
    for i in range(CRAWLER_ANSWERS_NUMBERS//5+1):
        slide()
    sleep(2)

    question_page_text = driver.page_source
    question_tree = etree.HTML(question_page_text)
    # 1.问题名
    question_title=question_tree.xpath('//h1[@class="QuestionHeader-title"]/text()')    # 问题名
    print(question_title)
    # 2.问题详情
    question_detail_list=question_tree.xpath('//div[contains(@class,"QuestionHeader")]//span[@id="content"]/span[@itemprop="text"]//text()')
    question_detail=''.join(question_detail_list)#问题详情
    # 3.回答详情
    answers_list=question_tree.xpath('//div[contains(@class,"Question-main")]//span[@id="content"]/span[@itemprop="text"]')
    answers_text_list=[] #回答详情
    answers_num=0
    for answer in answers_list:
        answers_num+=1
        if answers_num>CRAWLER_ANSWERS_NUMBERS:
            break
        paragraph=answer.xpath('./p//text()')
        answers_text=''.join(paragraph)
        answers_text_list.append(answers_text)
    print("共收集到",len(answers_text_list),"条回答")
    data={
        "question_title":question_title,
        "question_detail":question_detail,
        "answers_list":answers_text_list,
    }
    data_list.append(data)
with open('ZhiHuData.csv','w+',newline='',encoding='utf-8') as f:
    filednames=["question_title","question_detail","answers_list"]
    writer=csv.DictWriter(f,filednames)
    writer.writeheader()
    writer.writerows(data_list)

