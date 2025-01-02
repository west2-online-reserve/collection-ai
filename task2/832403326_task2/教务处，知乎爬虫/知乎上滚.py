from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
from selenium.webdriver.common.keys import Keys
import random

# 初始化 WebDriver

opt = Options()
opt.add_argument("--disable-blink-features=AutomationControlled")
web = Chrome(options=opt)

# 打开知乎首页
web.get('https://www.zhihu.com/')
web.implicitly_wait(100)

# 搜索关键词“庄子”
search_box = web.find_element(By.XPATH, '//*[@id="Popover1-toggle"]')
search_box.send_keys('庄子', Keys.ENTER)
web.implicitly_wait(5)

# 筛选为“只看问答”
web.find_element(By.XPATH, '//*[@id="root"]/div/main/div/div[1]/div/div/div').click()  # 筛选按钮
web.implicitly_wait(5)
web.find_element(By.XPATH, '//*[@id="root"]/div/main/div/div[1]/div[2]/ul[1]/li[2]/div').click()  # 只看问答
web.implicitly_wait(5)

# 滚动加载问题，爬取 50 个问题
questions = []  # 用于存储问题的标题和链接
while len(questions) < 2:
    web.execute_script("window.scrollTo(0, document.body.scrollHeight);")#模拟JS操作，y方向滚动到最底部
    time.sleep(2)
    # 获取所有问题标题和链接
    elements = web.find_elements(By.XPATH, '//h2[@class="ContentItem-title"]/span/div/div/div/a')
    for elem in elements:
        try:
            title = elem.text
            link = elem.get_attribute('href')
            if link not in [q['link'] for q in questions]:  # 防止重复
                questions.append({'title': title, 'link': link})
        except Exception as e:
            print(f"获取问题失败: {e}")
    print(f"已获取 {len(questions)} 个问题")
    if len(questions) >= 2:
        break

# 转换为 DataFrame
df_questions = pd.DataFrame(questions)

# 打印前 5 个问题
print(df_questions.head())

# 保存为 CSV 文件
df_questions.to_csv("zhihu_questions.csv", index=False, encoding='utf-8')



# 爬取回答
answers = []
for idx, row in df_questions.iterrows():
    print(f"正在爬取第 {idx + 1} 个问题：{row['title']}")
    web.get(row['link'])  # 打开问题链接
    wait = WebDriverWait(web, 30)  # 设置显式等待的超时时间为 30 秒

    # 点击“查看全部回答”按钮
    try:
        view_all_button = wait.until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="root"]/div/main/div/div/div[3]/div[1]/div/div[1]/a'))
        )
        view_all_button.click()  # 点击按钮
        print("点击‘查看全部回答’按钮成功")
    except Exception as e:
        print("没有找到‘查看全部回答’按钮，可能已经加载所有回答")

   
   


# 模拟上下滚动的逻辑
    previous_count = 0
    scroll_count = 0
    max_scroll = 50  # 设置最大滚动次数

    while scroll_count < max_scroll:
        # 正常向下滚动
        web.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(1.5, 3))  # 随机等待 1.5~3 秒，模拟真实用户操作

        # 随机上滚一小段
        if random.random() > 0.7:  # 30% 的几率触发上滚
            web.execute_script("window.scrollBy(0, -500);")  # 上滚 500 像素
            time.sleep(random.uniform(1, 2))  # 短暂等待再继续

        try:
            # 等待新回答加载
            wait.until(lambda driver: len(driver.find_elements(By.XPATH, '//div[@class="RichContent-inner"]')) > previous_count)
        except:
            print("没有检测到新回答加载，可能已经到底")
            break

        # 更新回答数量
        current_count = len(web.find_elements(By.XPATH, '//div[@class="RichContent-inner"]'))
        if current_count == previous_count:
            print("回答数量不再增加，停止滚动")
            break

        previous_count = current_count
        scroll_count += 1
        print(f"滚动次数: {scroll_count}, 当前回答数: {current_count}")

    # 提取回答内容
    answer_elements = web.find_elements(By.XPATH, '//div[@class="RichContent-inner"]')
    for answer_elem in answer_elements[:20]:  # 只提取前 20 条回答
        try:
            content = answer_elem.text  # 提取回答文本
            answers.append({'question': row['title'], 'link': row['link'], 'answer': content})
        except Exception as e:
            print(f"获取回答失败: {e}")

# 转换为 DataFrame 并保存
df_answers = pd.DataFrame(answers)
df_answers.to_csv("zhihu_answers.csv", index=False, encoding="utf-8")
print("回答数据已保存！")