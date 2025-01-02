from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

# 配置浏览器选项
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
question_data = []  # 用于存储问题标题和链接
seen_links = set()  # 用集合来记录已经处理过的链接
while len(question_data) < 2:
    web.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    # 获取所有问题标题和链接
    questions = web.find_elements(By.XPATH, '//h2[@class="ContentItem-title"]/span/div/div/div/a')
#//*[@id="SearchMain"]/div/div/div/div[2]/div/div/div/h2/span/div/div/div/a/span   
    for question in questions:
        try:
            title = question.text
            link = question.get_attribute('href')
            # 判断链接是否已经出现过，如果没有就添加
            if link not in seen_links:
                seen_links.add(link)  # 记录已经访问的链接
                question_data.append({'title': title, 'link': link})
            # 如果已经收集到50个问题，直接停止
            if len(question_data) >= 2:
                break
        except Exception as e:
            print(f"获取问题失败: {e}")
    
    print(f"已获取 {len(question_data)} 个问题")

# 打印结果，检查是否已获取到50个问题
print(f"最终获取了 {len(question_data)} 个问题")

# 爬取每个问题的前 20 条回答
answer_data = []  # 用于存储答案数据
for idx, question in enumerate(question_data[:2]):
    print(f"正在爬取第 {idx+1} 个问题: {question['title']}")
    web.get(question['link'])
    time.sleep(3)
    # 滚动加载答案
    for _ in range(5):  # 假设滚动 5 次可以加载 20 个回答
        web.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
    # 获取答案
    answers = web.find_elements(By.XPATH, '//div[@class="RichContent-inner"]')
    for answer in answers[:20]:  # 只取前 20 条回答
        try:
            content = answer.text
            answer_data.append({
                'question': question['title'],
                'link': question['link'],
                'answer': content
            })
        except Exception as e:
            print(f"获取回答失败: {e}")

# 保存结果到 CSV
df_questions = pd.DataFrame(question_data)
df_answers = pd.DataFrame(answer_data)
df_questions.to_csv('zhihu_questions.csv', index=False, encoding='utf-8')
df_answers.to_csv('zhihu_answers.csv', index=False, encoding='utf-8')

print("爬取完成，数据已保存！")