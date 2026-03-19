from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver # type hint
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webelement import WebElement

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.common.exceptions import NoSuchElementException

import time

general_list = []

import csv

def write_questions_to_csv(data, filename=r'west2_online\web_crawler\selenium\questions.csv'):
    """
    将字典列表写入 CSV 文件，格式如下：
    第一行：列头（空, 问题描述, 回答1, 回答2, ...）
    后续行：行头（问题1, 问题2, ...），问题描述，以及每个问题的回答列表。

    :param data: 列表，每个元素是一个字典，必须包含两个键：
                 '问题描述'（字符串）和 '回答列表'（列表）
    :param filename: 输出的 CSV 文件名（默认 questions.csv）

    reference from ai
    """
    if not data:
        print("数据为空，无法生成 CSV")
        return

    # 1. 确定最大回答数，用于生成列头
    max_answers = max(len(item['回答列表']) for item in data)

    # 2. 准备列头（第一行）
    # 第一列留空（作为行表头列），第二列为“问题描述”，后面为“回答1”到“回答N”
    header = ['问题标题'] + ['问题描述'] + [f'回答{i}' for i in range(1, max_answers + 1)]

    # 3. 准备数据行
    rows = []
    for idx, item in enumerate(data, start=1):
        row = [item['问题标题']]                # 行表头：问题1, 问题2, ...
        row.append(item['问题描述'])         # 问题描述
        answers = item['回答列表']
        # 填充回答，不足 max_answers 的部分用空字符串补齐
        answers_padded = answers + [''] * (max_answers - len(answers))
        row.extend(answers_padded)
        rows.append(row)

    # 4. 写入 CSV 文件
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"CSV 文件已生成：{filename}")


def grab_answers(driver:WebDriver,question_element:WebElement,general_list:list,title:str):
    # 对一个问题爬取10条回答
    ANSWER_NUM = 10

    # 获得问题描述
    # -等待页面载入完成
    time.sleep(4)
    # -点击显示全部
    try:
        button = driver.find_element(By.CSS_SELECTOR,".QuestionRichText.QuestionRichText--collapsed")
        button.find_element(By.TAG_NAME,"button").click()
    except NoSuchElementException:
        pass

    # -获得文本
    try:
        question_detail = driver.find_element(By.CLASS_NAME,"QuestionRichText")\
                                .find_element(By.ID,"content")\
                                .find_element(By.TAG_NAME,'span').text
    except NoSuchElementException:
        question_detail = '无'

    # 获得回答文本
    # - 滚动出n条回答
    ele_list = driver.find_elements(By.CLASS_NAME,'List-item')
    len_page_old = -1
    len_page_new = len(ele_list)
    while len(ele_list) < ANSWER_NUM and len_page_old != len_page_new:
        # 向下滚动 500 像素
        driver.execute_script("window.scrollBy(0, 500);")
        ele_list = driver.find_elements(By.CLASS_NAME,'List-item')
        len_page_old = len_page_new
        len_page_new = len(ele_list)
        
    time.sleep(4)
    ele_list = driver.find_elements(By.CLASS_NAME,'List-item')
    # - 获得文本
    ans_list = []
    for ans_count in range(0,len_page_new if len_page_new < ANSWER_NUM else ANSWER_NUM):
        ans_text = ele_list[ans_count].find_element(By.CLASS_NAME,"RichContent-inner")\
                                    .find_element(By.ID,'content').text
        ans_list.append(ans_text)
    
    # 存入字典,格式：{"问题描述":str,"回答列表":list[str]}
    question_dict = {
        '问题标题':title,
        '问题描述':question_detail,
        '回答列表':ans_list
    }

    # 存入表
    general_list.append(question_dict)

    # 关闭标签页，跳回原页面
    driver.close()
    driver.switch_to.window(driver.window_handles[0])


if __name__ == '__main__':
    # 配置浏览器和驱动路径(手动，可以默认)
    from selenium.webdriver.edge.service import Service
    service = Service(executable_path=r"D:\FOREFOX\Softs\chromedriver-win64\chromedriver.exe")

    # 浏览器配置
    options = Options()
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver:WebDriver = webdriver.Chrome(service=service,options=options)

    # 打开浏览器
    driver.get(r"https://www.zhihu.com/topic/19555513/unanswered")
    driver.implicitly_wait(5)

    # 载入cookies
    with open(r"west2_online\web_crawler\selenium\cookies.json",'r') as f:
        import json
        cookie_list = json.load(f)

    for cookie_dict in cookie_list:
        driver.add_cookie(cookie_dict)

    driver.refresh()

    # 滚动至有二十个问题
    QUESION_NUM = 20
    element_list = driver.find_elements(By.CSS_SELECTOR,".List-item.TopicFeedItem")
    while len(element_list) < QUESION_NUM:
        # 向下滚动 500 像素
        driver.execute_script("window.scrollBy(0, 500);")
        element_list = driver.find_elements(By.CSS_SELECTOR,".List-item.TopicFeedItem")

    time.sleep(4)
    element_list = driver.find_elements(By.CSS_SELECTOR,".List-item.TopicFeedItem")
    for question_count in range(0,QUESION_NUM):
        # -定位问题
        ele = element_list[question_count].find_element(By.CLASS_NAME,"QuestionItem-title")\
                                        .find_element(By.TAG_NAME,'a')

        # -获取问题标题
        question_title = ele.text

        # -点击问题
        action = ActionChains(driver)
        action.click(ele).perform()
        driver.switch_to.window(driver.window_handles[-1])

        # -获得问题描述和回答
        grab_answers(driver,ele,general_list,question_title)

    # 写入文件
    write_questions_to_csv(general_list)

    time.sleep(10)
    driver.quit()
