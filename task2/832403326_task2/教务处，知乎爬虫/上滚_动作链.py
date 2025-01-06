from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

import pandas as pd
import time
import random
from selenium.webdriver.common.keys import Keys

# 初始化 WebDriver
opt = Options()
opt.add_argument("--disable-blink-features=AutomationControlled")
web = Chrome(options=opt)

# 打开知乎首页
web.get("https://www.zhihu.com/")
web.implicitly_wait(10)

# 搜索关键词“庄子”
search_box = web.find_element(By.XPATH, '//*[@id="Popover1-toggle"]')
search_box.send_keys("庄子", Keys.ENTER)
web.implicitly_wait(5)

# 筛选为“只看问答”
web.find_element(
    By.XPATH, '//*[@id="root"]/div/main/div/div[1]/div/div/div'
).click()  # 筛选按钮
web.implicitly_wait(5)
web.find_element(
    By.XPATH, '//*[@id="root"]/div/main/div/div[1]/div[2]/ul[1]/li[2]/div'
).click()  # 只看问答
web.implicitly_wait(5)

# 滚动加载问题，爬取 2 个问题
questions = []  # 用于存储问题的标题和链接
links_set = set()  # 用于存储链接以防重复

questions = []  # 用于存储问题的标题和链接
links_set = set()  # 用于存储链接以防重复

while True:
    web.execute_script(
        "window.scrollTo(0, document.body.scrollHeight);"
    )  # 模拟滚动到底部
    time.sleep(2)  # 等待页面加载新内容

    # 获取所有问题标题和链接
    elements = web.find_elements(
        By.XPATH, '//h2[@class="ContentItem-title"]/span/div/div/div/a'
    )
    for elem in elements:
        try:
            title = elem.text
            link = elem.get_attribute("href")
            if link not in links_set:  # 防止重复
                questions.append({"title": title, "link": link})
                links_set.add(link)
        except Exception as e:
            print(f"获取问题失败: {e}")

    # 如果问题数量达到目标数量，停止滚动
    if len(questions) >= 50:
        questions = questions[:50]  # 截取前 2 个问题
        break

print(f"已获取 {len(questions)} 个问题")
# 转换为 DataFrame
df_questions = pd.DataFrame(questions)

# 打印问题信息
print(df_questions.head())

# 保存问题到 CSV 文件
df_questions.to_csv("zhihu_questions.csv", index=False, encoding="utf-8")

# 爬取每个问题的回答

# 爬取每个问题的回答
answers = []  # 用于存储回答内容
for idx, row in df_questions.iterrows():
    print(f"正在爬取第 {idx + 1} 个问题：{row['title']}")
    web.get(row["link"])  # 打开问题链接
    wait = WebDriverWait(web, 30)  # 设置显式等待的超时时间为 30 秒

    # 点击“查看全部回答”按钮
    try:
        view_all_button = wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="root"]/div/main/div/div/div[3]/div[1]/div/div[1]/a',
                )
            )
        )
        view_all_button.click()  # 点击按钮
        print("点击‘查看全部回答’按钮成功")
    except Exception as e:
        print("没有找到‘查看全部回答’按钮，可能已经加载所有回答")

    # 初始化动作链
    action = ActionChains(web)
    previous_height = 0  # 页面之前的高度
    scroll_count = 0
    max_scroll = 50  # 最大滚动次数
    answers_set = set()  # 用于存储去重的回答内容

    while scroll_count < max_scroll:
        # 向下滚动到底部
        web.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)  # 短暂停留，等待加载

        # 向上滚动一小段距离（上滑 500 像素）
        action.scroll_by_amount(0, -500).perform()
        time.sleep(1)  # 短暂停留

        # 再次向下滚动到底部
        web.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)  # 再次停顿

        # 检测页面高度变化
        current_height = web.execute_script("return document.body.scrollHeight")
        if current_height == previous_height:
            print("页面高度未变化，停止滚动")
            break
        previous_height = current_height

        # 提取回答内容
        answer_elements = web.find_elements(
            By.XPATH, '//div[@class="RichContent-inner"]'
        )
        for answer_elem in answer_elements:
            try:
                content = answer_elem.text
                if content not in answers_set:  # 检查去重
                    answers.append(
                        {
                            "question": row["title"],
                            "link": row["link"],
                            "answer": content,
                        }
                    )
                    answers_set.add(content)
                    if len(answers_set) >= 20:  # 如果已获取 20 个回答，停止爬取
                        print("已爬取 20 个回答，停止滚动")
                        break
            except Exception as e:
                print(f"获取回答失败: {e}")

        # 如果已爬取 20 个回答，跳出滚动循环
        if len(answers_set) >= 20:
            break

        scroll_count += 1
        print(f"滚动次数: {scroll_count}, 当前回答数: {len(answers_set)}")

print(f"共爬取到 {len(answers)} 条回答")

# 转换为 DataFrame 并保存
df_answers = pd.DataFrame(answers)
df_answers.to_csv("zhihu_answers.csv", index=False, encoding="utf-8")
print("回答数据已保存！")
