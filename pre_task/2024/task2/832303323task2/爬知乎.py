from selenium import webdriver
import os
import time
import csv
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

# 获取桌面路径
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

# 创建CSV文件路径
csv_file_path = os.path.join(desktop_path, 'zhihu.csv')

# 打开CSV文件进行写入
with open(csv_file_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    
    # 加上反爬伪装 能够爬的更多一些
    options = Options()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument('--disable-gpu')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=options)
    actions = ActionChains(driver)

    # 加载指定的页面
    driver.get('https://www.zhihu.com/')
    time.sleep(20)  # 这里手动登录

    driver.get('https://www.zhihu.com/topic/19554298/top-answers')
    driver.maximize_window()
    time.sleep(5)

    # 滚动页面以加载更多内容
    for _ in range(1):
        js = "window.scrollTo(0,document.body.scrollHeight)"
        driver.execute_script(js)
        time.sleep(2)
    
    time.sleep(5)

    # 拉到底 获得尽量多的问题
    question1 = driver.find_elements(By.XPATH, "//h2/div/a")
    url_list = []
    title_list = []
    Answer = []
    question_content = []

    # 问题的爬取
    for question in question1:
        url_list.append(question.get_attribute("href"))
        title_list.append(question.text)
    
    # 写入标题行
    writer.writerow(title_list)

    # 爬取答案
    for url in url_list:
        driver.get(f"{url}")
        time.sleep(5)
        
        # 是否有展开的按钮
        try:
            stretch = driver.find_element(By.XPATH, '//div[@class="QuestionRichText QuestionRichText--expandable QuestionRichText--collapsed"]/div/button').click()
            question_content.append(driver.find_element(By.XPATH, '//div[@class="QuestionRichText QuestionRichText--expandable"]').text)
        except:
            try:
                question_content.append(driver.find_element(By.XPATH, '//div/span[@class="RichText ztext css-ob6uua"]').text)
            except:
                question_content.append("没有内容")
        
        time.sleep(5)
        
        try:
            driver.find_element(By.XPATH, '//div[@class="Card ViewAll"]/a').click()
            time.sleep(1)
        except:
            pass

        # 模拟用户滚动行为
        for i in range(20):
            js_real = f"window.scrollTo(0, {i*5000});"
            driver.execute_script(js_real)
            actions.send_keys(Keys.PAGE_UP).perform()
            actions.send_keys(Keys.PAGE_DOWN).perform()
            time.sleep(1)
        
        # 找到问题答案
        answers = driver.find_elements(By.XPATH, '//div[@class="ContentItem AnswerItem"]')
        answer_text = [element.text for element in answers]
        Answer.append(answer_text)
        print(Answer)

    # 写入问题内容和答案
    writer.writerow(question_content)
    for answer in Answer:
        writer.writerow(answer)

    # 关闭浏览器
    driver.quit()