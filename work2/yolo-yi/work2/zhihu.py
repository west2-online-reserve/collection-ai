import time
import csv
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

#互联网话题
TARGET_TOPIC_URL = "https://www.zhihu.com/topic/19550517/top-answers"
CSV_FILENAME = "zhihu_data.csv"
MAX_QUESTIONS = 20
MAX_ANSWERS_PER_QUESTION = 10

def init_driver():
    #启动同目录下的ChromeDriver
    current_dir = os.path.dirname(os.path.abspath(__file__))
    driver_name = 'chromedriver.exe'
    chromedriver_path = os.path.join(current_dir, driver_name)

    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
            })
        """
    })
    return driver

def login_zhihu(driver):
    print("正在打开知乎登录页")
    driver.get("https://www.zhihu.com/signin")
    print("-" * 30)
    print("请在浏览器中扫码登录。")
    print("登录成功后，按下 [Enter] 键继续...")
    print("-" * 30)
    input()
    print("检测到已确认登录，开始执行爬虫任务")

def scroll_to_bottom(driver, speed=2):
    driver.execute_script("window.scrollBy(0, 500);")
    time.sleep(speed)

def scrape_topic_questions(driver, topic_url):
    #一：抓取链接和标题
    driver.get(topic_url)
    time.sleep(3)
    #用一个列表来存储元组 (url, title)
    questions_data_list = []
    seen_urls = set()
    print(f"正在话题页抓取问题链接，目标: {MAX_QUESTIONS} 个...")

    while len(questions_data_list) < MAX_QUESTIONS:
        #得到一个所有问题列表
        elements = driver.find_elements(By.CSS_SELECTOR, ".ContentItem-title a")
        #遍历每一个问题
        for elem in elements:
            try:
                url = elem.get_attribute("href")
                title = elem.text
                # 只有当它是问题链接，且之前没抓过时，才添加
                if "/question/" in url and url not in seen_urls:
                    seen_urls.add(url)
                    # 将 URL 和 Title 打包存入列表
                    questions_data_list.append((url, title))

                    if len(questions_data_list) >= MAX_QUESTIONS:
                        break
            except Exception:
                continue  # 防止某个元素抓取报错中断循环
        print(f"当前已收集 {len(questions_data_list)} 个问题链接...")
        if len(questions_data_list) < MAX_QUESTIONS:
            scroll_to_bottom(driver)
        else:
            break
    return questions_data_list

def scrape_details(driver, questions_data_list):
    all_data = []

    for index, (q_url, q_title) in enumerate(questions_data_list):
        print(f"[{index + 1}/{len(questions_data_list)}] 正在处理: {q_title}")
        driver.get(q_url)
        time.sleep(2)
        try:
            view_all_btn = driver.find_element(By.CSS_SELECTOR, "a.ViewAll-QuestionMainAction")
            if view_all_btn.is_displayed():
                print("检测到回答被折叠，点击“查看全部”")
                driver.execute_script("arguments[0].click();", view_all_btn)
                time.sleep(4)
        except:
            # 如果没有“查看全部”按钮，直接跳过即可，不影响后续流程
            pass
        # 获取当前知乎问题的详细描述
        detail = "无详细描述"
        try:
            try:
                expand_btn = driver.find_element(By.CSS_SELECTOR, "button.QuestionRichText-more")
                if expand_btn.is_displayed():
                    driver.execute_script("arguments[0].click();", expand_btn)
                    time.sleep(0.5)
            except:
                # 跳过
                pass

            detail_elem = driver.find_element(By.CSS_SELECTOR, ".QuestionRichText span[itemprop='text']")
            if detail_elem.text.strip():
                detail = detail_elem.text
        except:
            # 忽略异常：如果没有详细描述元素，使用默认值 "无详细描述"
            pass
        #获取回答
        answers_combined = "无回答信息"
        try:
            found_answers = []
            scroll_attempts = 0
            max_scroll_attempts = 15  # 增加尝试次数上限
            # 初始定位
            found_answers = driver.find_elements(By.CSS_SELECTOR, ".List-item .RichContent")
            # 循环：只要回答不足 10 个，且尝试次数没用完，就一直滚
            while len(found_answers) < MAX_ANSWERS_PER_QUESTION and scroll_attempts < max_scroll_attempts:
                driver.execute_script("window.scrollBy(0, 1500);")
                time.sleep(2)
                # 再次检查数量
                found_answers = driver.find_elements(By.CSS_SELECTOR, ".List-item .RichContent")
                scroll_attempts += 1
                # 如果滑到了底还是不够，可能尝试一次“瞬移到底部”作为补救
                if scroll_attempts % 3 == 0:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
            # 循环结束，开始提取文本
            print(f"滚动结束，最终获取到 {len(found_answers)} 条回答。开始提取...")

            answers_text = []
            count = 0
            for item in found_answers:
                if count >= MAX_ANSWERS_PER_QUESTION:
                    break
                try:
                    text_elem = item.find_element(By.CSS_SELECTOR, ".RichText")
                    ans_text = text_elem.text.replace('\n', ' ').strip()
                    if ans_text:
                        answers_text.append(f"【回答{count + 1}】: {ans_text[:200]}...")
                        count += 1
                except:
                    continue

            if answers_text:
                answers_combined = " || ".join(answers_text)
        except Exception as e:
            print(f"  -> 抓取回答出错: {e}")

        all_data.append([q_title, detail, answers_combined])

    return all_data

def save_to_csv(data):
    #保存数据到
    headers = ["问题名", "问题具体内容", "回答信息"]
    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"爬取结束，数据已保存至 {CSV_FILENAME}")

#主程序
if __name__ == "__main__":
    driver = init_driver()
    try:
        # 1. 登录
        login_zhihu(driver)
        # 2. 获取问题列表
        questions_data = scrape_topic_questions(driver, TARGET_TOPIC_URL)
        # 3. 获取详细内容(问题名，问题详细内容，回答)
        data = scrape_details(driver, questions_data)
        # 4. 保存
        save_to_csv(data)
    finally:
        time.sleep(3)
        driver.quit()