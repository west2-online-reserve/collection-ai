from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.chrome.service import Service
import json
import time
import csv
import re

from selenium.webdriver.common.by import By

pattern = re.compile(r'<[^>]+>') #匹配html标签


def find_answer(url,count=10) -> list:
    nodes = []
    answers = []
    temp_height = 0
    while True:
        if len(nodes) < count:
            #向下滑动加载
            driver.execute_script('window.scrollBy(0,5000)')
            time.sleep(0.5)
            check_height = driver.execute_script(
                "return document.documentElement.scrollTop || window.pageYOffset || document.body.scrollTop;")
            if check_height == temp_height:
                #页面到底
                break
            temp_height = check_height
            time.sleep(1)
            try:
                nodes = driver.find_elements('xpath',
                                             '//*[@id="QuestionAnswers-answers"]/div/div/div/div[2]/div/div[@class="List-item"]/div/div[@class="ContentItem AnswerItem"]'
                                             )
            except BaseException as e:
                print(e)
                input('回车继续')
                continue
        else:
            break
    for answer in nodes:
        try:
            rich_text_html = answer.find_element('xpath',
                                                 './div[@class="RichContent RichContent--unescapable"]/span[1]').get_attribute(
                'innerHTML')
            answer_content = re.sub(pattern, '', rich_text_html)  # 通过正则表达式替换所有html标签,只留下文本
            answers.append(answer_content)
        except BaseException as e:
            print(e)
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    print(answers)
    return answers



def main():
    writer = csv.writer(open('zhihu.csv', 'w', encoding='utf-8', newline=''))
    writer.writerow(['标题', '问题', '回答'])
    # 读取cookies
    with open("cookies.json", 'r', encoding='utf-8') as f:
        cookies = json.loads(f.read())

    driver.get("https://www.zhihu.com/topic/20063922/hot")

    # 先加载网页才能添加cookies
    for cookie in cookies:
        driver.add_cookie(cookie)
    # 访问话题页面
    driver.refresh()
    time.sleep(1)
    # 向下滚动5000个像素 加载更多问题
    driver.execute_script('window.scrollBy(0,5000)')
    time.sleep(2)
    node_list = driver.find_element("xpath", '//*[@id="TopicMain"]/div[4]/div[2]/div/div')
    items = node_list.find_elements("xpath",
                                    './div[@class="List-item TopicFeedItem"]/div/div[@class="ContentItem AnswerItem"]')
    items_data = []
    for i in items:
        url = i.find_element('xpath','./h2/div/meta[@itemprop="url"]').get_attribute('content')
        driver.execute_script(f'window.open("{url}")')
        driver.switch_to.window(driver.window_handles[1])
        title = ''
        try:

            title = driver.find_element('xpath',
                                        '//*[@id="root"]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/h1').text
            try:
                #展开全部按钮
                driver.find_element('xpath',
                                    '//div[@class="QuestionRichText QuestionRichText--expandable QuestionRichText--collapsed"]/div/button').click()
            finally:
                qus = driver.find_element('xpath',
                                          '//div[@class="QuestionRichText QuestionRichText--expandable"]').get_attribute(
                    'innerHTML')

            # 去除html标签
            question = re.sub(pattern, '', qus)
        except NoSuchElementException:
            question = ''
        answers = find_answer(url)

        answer_text = ''
        for b in range(len(answers)):
            answer_text += str(b)+'.'+' ||||| '+ answers[b] + ' ||||| '
        writer.writerow([title, question, answer_text])

        items_data.append([title, question, answers])
    print(items_data)

    input()



if __name__ == '__main__':
    options = webdriver.ChromeOptions()
    # 设置chrome位置
    options.binary_location = r"D:\learn-AI\learn-AI\task2\zhihu\chrome\chrome.exe"
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-blink-features=AutomationControlled")

    # 设置webdriver服务
    service = Service(r"D:\learn-AI\learn-AI\task2\zhihu\chrome\chromedriver.exe")

    driver = webdriver.Chrome(service=service, options=options)
    # with open('stealth.min.js', encoding='utf-8') as f: #绕过检测js
    # driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': f.read()})
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                    })
                """
        },
    )
    main()