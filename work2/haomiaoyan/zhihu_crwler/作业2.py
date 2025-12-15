# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import random

# 配置区域
TOPIC_URL = "https://www.zhihu.com/topic/19551275/hot"
OUTPUT_CSV = "zhihu_topic_data.csv"
TARGET_QUESTION_COUNT = 20
ANSWERS_PER_QUESTION = 10


# 辅助函数：检测验证码
def check_verification(driver):
    """检测是否弹出验证码或安全防火墙"""
    verification_keywords = ["安全验证", "验证码", "Captcha", "unusual traffic"]
    if any(k in driver.page_source for k in verification_keywords):
        print("\n[警告] 触发了知乎验证码！完成验证后按回车...")
        input(">>> 完成后请按回车: ")
        return True
    return False


# 初始化浏览器
options = EdgeOptions()
options.add_argument("--start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Edge(options=options)

# 登录知乎
driver.get("https://www.zhihu.com/signin")
print("请在 30 秒内完成扫码...")
time.sleep(30)

# 进入话题页
driver.get(TOPIC_URL)
time.sleep(5)

# 链接预处理（加载、去重、过滤）
print("正在加载话题列表并预处理链接...")
unique_question_urls = set()
MAX_LINK_SCROLLS = 10  # 在话题页最多滚动 10 次
scroll_count = 0

while len(unique_question_urls) < TARGET_QUESTION_COUNT and scroll_count < MAX_LINK_SCROLLS:
    check_verification(driver)

    links = driver.find_elements(By.CSS_SELECTOR, 'a[data-za-detail-view-element_name="Title"]')

    for q in links:
        q_url = q.get_attribute("href")

        if "zhihu.com/question/" in q_url and "zhuanlan.zhihu.com" not in q_url:
            if "/answer/" in q_url:
                q_url = q_url.split("/answer/")[0]

            unique_question_urls.add(q_url)

    print(f"当前已收集到 {len(unique_question_urls)} 个不重复问题链接...")

    if len(unique_question_urls) < TARGET_QUESTION_COUNT:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        scroll_count += 1

# 转换为 list，并只取目标数量
question_list = list(unique_question_urls)[:TARGET_QUESTION_COUNT]
print(f"最终确定 {len(question_list)} 个不重复问题进行爬取。")
# =========================================================

questions_data = []
processed_questions = 0

# 主循环：遍历去重后的问题列表
for q_url in question_list:
    try:
        processed_questions += 1
        print(f"\n[{processed_questions}/{TARGET_QUESTION_COUNT}] 访问：{q_url}")

        driver.execute_script("window.open(arguments[0])", q_url)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(4)
        check_verification(driver)

        # 抓取基本信息
        try:
            title = driver.find_element(By.TAG_NAME, "h1").text
            desc = driver.find_element(By.CLASS_NAME, "QuestionRichText").text
        except:
            title, desc = "无标题", ""

        # 模拟回弹滚动
        temp_answers = set()
        scroll_attempts = 0

        while len(temp_answers) < ANSWERS_PER_QUESTION and scroll_attempts < 20:  # 增加尝试次数上限
            # 向下滚动到底部 (触发加载)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2 + random.random())  # 增加随机等待时间

            # 向上回弹一段距离 (模拟用户阅读行为)
            # 向上滚动 800-1200 像素，模拟回看
            driver.execute_script(f"window.scrollBy(0, -{random.randint(800, 1200)});")
            time.sleep(1 + random.random())

            # 重新向下滚动 (确保新的内容进入视口)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2 + random.random())

            # 抓取当前页面所有回答
            answer_cards = driver.find_elements(By.CSS_SELECTOR, ".ContentItem.AnswerItem")
            for card in answer_cards:
                try:
                    rich_text_element = card.find_element(By.CSS_SELECTOR, ".RichText.ztext")
                    content = rich_text_element.text.strip()

                    if content and len(content) > 5:
                        temp_answers.add(content)
                        if len(temp_answers) >= ANSWERS_PER_QUESTION:
                            break
                except:
                    continue

            scroll_attempts += 1
            # 实时输出进度
            print(f"  - 已尝试 {scroll_attempts} 次回弹滚动，当前抓取到 {len(temp_answers)} 条回答...")

            # 如果回答数量足够，立即跳出外部循环
            if len(temp_answers) >= ANSWERS_PER_QUESTION:
                break

        # 存储数据
        for text in temp_answers:
            questions_data.append({
                "类型": "问题",
                "标题": title,
                "内容": desc,
                "回答文本": text
            })

        print(f"最终抓取 {len(temp_answers)} 条回答")
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(1 + random.random())  # 增加随机休息时间

    except Exception as e:
        print(f"处理问题 {q_url} 时出错: {e}")
        if len(driver.window_handles) > 1:
            driver.close()
        driver.switch_to.window(driver.window_handles[0])
        continue

# 保存
if questions_data:
    df = pd.DataFrame(questions_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n任务完成！总数据：{len(df)} 条")
    print(f"实际抓取不同问题数: {len(question_list)}")
    print(f"数据已保存至: {OUTPUT_CSV}")
else:
    print("未抓取到任何数据。")

driver.quit()
