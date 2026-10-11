from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import os
import json
import re


# 获取用户信息的路径
base_dir = os.path.dirname(os.path.abspath(__file__))
profile_path = os.path.join(base_dir, "chrome_profile")
information_storage_path = os.path.join(base_dir, "20 个问答贴的回答.txt")

options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
# 自动存入和读取登录的信息
options.add_argument(f"user-data-dir={profile_path}")

wd = webdriver.Chrome(options=options)
wait = WebDriverWait(wd, 10)

# 获取问答页标题的标签为 type 的内容
def get_question_title_one_information(type):
    element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f'meta[itemprop="{type}"]')))
    return element.get_attribute('content')

# 获取标题的相关信息
def research_question_title_information():
    title =  get_question_title_one_information('name')
    url = get_question_title_one_information('url')
    key_words = get_question_title_one_information('keywords')
    answer_count = get_question_title_one_information('answerCount')
    return {
        'title': title, 'url': url,
        'key_word': key_words, 'answer_count':answer_count}

# 通过会回答的原始元素获得正文内容
def research_detail_information(raw_element) -> str:
    text_elements = raw_element.find_elements(By.CSS_SELECTOR, 'p[data-pid]')
    piece_list = [elem.text for elem in text_elements]
    text = "\r\n".join(piece_list)
    return text

# 获取 num 个回答的所有内容，返回成列表
def get_answer_list(num) -> list:
    raw_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[class="List-item"][tabindex="0"]')))
    while len(raw_elements) < num:
        wd.find_element(By.TAG_NAME,"body").send_keys(Keys.PAGE_DOWN)
        time.sleep(0.4)
        raw_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[class="List-item"][tabindex="0"]')))
    answer_list = []
    current_num = 1
    for raw_element in raw_elements:
        if current_num > num:
            break
        answer_list.append(research_detail_information(raw_element))
        current_num += 1
    return answer_list

# 获取问答的所有信息，封装成一个字典
def get_qs_information(url):
    wd.get(url)
    qs_dict:dict = research_question_title_information()
    answer_list = get_answer_list(10)
    qs_dict["answer"] = answer_list
    return qs_dict

# 通过原始的问答贴元素，提取具体的 url
def get_qs_url(raw_element):
    url_element = raw_element.find_element(By.CSS_SELECTOR, 'a[target="_blank"][data-za-detail-view-element_name="Title"]')
    raw_url = url_element.get_attribute('href')
    pattern = re.compile(r'(.*?)/answer.*?')
    url = re.findall(pattern, raw_url)[0]
    return url

# 获取 num 个问答贴的 url
def get_all_qs_url(num):
    selector_str = 'div[class="Card TopstoryItem TopstoryItem-isRecommend"][tabindex="0"]'
    raw_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector_str)))
    
    while len(raw_elements) < num:
        wd.find_element(By.TAG_NAME,"body").send_keys(Keys.PAGE_DOWN)
        time.sleep(0.4)
        raw_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector_str)))

    current_num = 1
    new_raw_elements = []
    for raw_element in raw_elements:
        if current_num > num:
            break
        new_raw_elements.append(get_qs_url(raw_element))
        current_num += 1
    
    return new_raw_elements


if __name__ == "__main__":

    # 登录
    wd.get("https://www.zhihu.com")
    input("登录完后，请输入回车以继续")

    # 获取 20 个问答贴的 url
    urls = get_all_qs_url(2)

    # 依次获取 20 个问答的的回答
    all_qs_information_list = []
    for url in urls:
        all_qs_information_list.append(get_qs_information(url))

    with open(information_storage_path, 'w', encoding="utf-8") as f:
        for item in all_qs_information_list:
            f.write(f"【标题】{item['title']}\r\n")
            f.write(f"链接：{item['url']}\r\n")
            f.write(f"关键词：{item['key_word']}\r\n")
            f.write(f"回答数量：{item['answer_count']}\r\n")
            f.write("------------------------------------------------------------\r\n")
            for idx,ans_text in enumerate(item["answer"], start=1):
                f.write(f"====回答{idx}====\r\n")
                f.write(ans_text)
                f.write("\r\n\r\n")

    print("文件存储完毕！")
    input("输入回车以关闭")
    wd.quit()
