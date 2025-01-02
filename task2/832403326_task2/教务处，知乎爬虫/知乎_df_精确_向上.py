from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# 配置浏览器选项
opt = Options()
opt.add_argument("--disable-blink-features=AutomationControlled")
web = Chrome(options=opt)

# 打开知乎首页
web.get('https://www.zhihu.com/')
wait = WebDriverWait(web, 100)
time.sleep(3)
# 搜索关键词“庄子”
search_box = web.find_element(By.XPATH, '//*[@id="Popover1-toggle"]')
search_box.send_keys('庄子', Keys.ENTER)

# 筛选为“只看问答”
try:
    filter_button = wait.until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="root"]/div/main/div/div[1]/div/div/div'))
    )
    filter_button.click()
    wait.until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="root"]/div/main/div/div[1]/div[2]/ul[1]/li[2]/div'))
    ).click()
except Exception as e:
    print("筛选失败：", e)

# 滚动加载问题，爬取 2 个问题
questions = []
links_set = set()

questions = []  # 用于存储问题的标题和链接
links_set = set()  # 用于存储链接以防重复

while True:
    web.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 模拟滚动到底部
    time.sleep(2)  # 等待页面加载新内容

    # 获取所有问题标题和链接
    elements = web.find_elements(By.XPATH, '//h2[@class="ContentItem-title"]/span/div/div/div/a')
    for elem in elements:
        try:
            title = elem.text
            link = elem.get_attribute('href')
            if link not in links_set:  # 防止重复
                questions.append({'title': title, 'link': link})
                links_set.add(link)
        except Exception as e:
            print(f"获取问题失败: {e}")

    # 如果问题数量达到目标数量，停止滚动
    if len(questions) >= 2:
        questions = questions[:2]  # 截取前 2 个问题
        break

print(f"已获取 {len(questions)} 个问题")
df_questions = pd.DataFrame(questions)
df_questions.to_csv("zhihu_questions.csv", index=False, encoding='utf-8')

answers = []
for idx, row in df_questions.iterrows():
    print(f"正在爬取第 {idx + 1} 个问题：{row['title']}")
    web.get(row['link'])
    answers_set = set()

    previous_count = 0
    scroll_count = 0
    max_scroll = 50
    while scroll_count < max_scroll:
        web.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        current_count = len(web.find_elements(By.XPATH, '//div[@class="RichContent-inner"]'))
        if current_count == previous_count:
            break
        previous_count = current_count
        scroll_count += 1

    answer_elements = web.find_elements(By.XPATH, '//div[@class="RichContent-inner"]')
    for answer_elem in answer_elements[:20]:
        try:
            content = answer_elem.text
            if content not in answers_set:
                answers.append({'question': row['title'], 'link': row['link'], 'answer': content})
                answers_set.add(content)
        except Exception as e:
            print(f"获取回答失败: {e}")

df_answers = pd.DataFrame(answers)
df_answers.to_csv("zhihu_answers.csv", index=False, encoding="utf-8")
print("回答数据已保存！")