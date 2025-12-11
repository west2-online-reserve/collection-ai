from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
import time
import re
import csv

csv_file = open('zhihu_data.csv', 'w', newline='', encoding='utf-8-sig')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['标题', '详情'] + [f'回答{i+1}' for i in range(10)])

service = EdgeService()

options = webdriver.EdgeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0')

options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)


driver = webdriver.Edge(service=service,options=options)
def main():

    try:

        driver.get('https://www.zhihu.com/topic/20106982/unanswered')
        print("等待登录...")
        input()
        time.sleep(2)

        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        links = driver.find_elements(By.XPATH,"//a[contains(@href, '/question/')]")

        print(f"找到 {len(links)} 个问题")
        hrefs = []
        for link in links[0:88:2]:
            hrefs.append(link.get_attribute('href'))

        for href in hrefs[::2]:
            try:
                driver.get(href)
                time.sleep(3)
                
                # 标题
                title = driver.title
                if " - 知乎" in title:
                    title = title.replace(" - 知乎", "")
                print(f"标题: {title}")
                
                # 展开
                try:
                    expand_btn = driver.find_element(By.XPATH, "//button[text()='显示全部 ']")  # 后面有个空格
                    expand_btn.click()
                    time.sleep(2)
                except:
                    pass
                
                # 提取详情
                try:
                    details = driver.find_element(By.CSS_SELECTOR, "div.QuestionRichText.QuestionRichText--expandable").get_attribute('outerText')
                except:
                    details = ""
                
                # 滚动
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(0.5)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                
                # 点击回答展开按钮
                for btn in driver.find_elements(By.XPATH, "//button[contains(text(), '展开')]")[:10]:
                    try:
                        btn.click()
                        time.sleep(0.5)
                    except:
                        pass

                answers = driver.find_elements(By.CSS_SELECTOR, 'div.List-item')
                
                # 写入CSV
                row = [clean_content(title), clean_content(details)]
                for i in range(10):
                    if i < len(answers):
                        text = clean_content(answers[i].get_attribute('outerText'))
                        row.append(text)
                    else:
                        row.append("")
                csv_writer.writerow(row)
                
                print(f"已保存")
                
            except Exception as e:
                print(f"出错: {e}")
                continue
            

    finally:
        driver.quit()
        csv_file.close()


def clean_content(text):
    """清理干扰信息"""
    if not text:
        return ""
    
    # 删除所有常见的干扰行
    patterns_to_remove = [
        r'赞同\s*\d+[\.\d万K]*',      # 赞同 356、赞同1.2万
        r'\d+\s*条评论',               # 44 条评论
        r'分享\s*\d*',                 # 分享、分享 12
        r'收藏\s*\d*',                 # 收藏、收藏 5
        r'喜欢\s*\d*',                 # 喜欢、喜欢 3
        r'收起\s*\d*',                 # 收起、收起​
        r'编辑于\s*.+',                # 编辑于 2023-12-12
        r'发布于\s*.+',                # 发布于 2023-12-12
        r'作者\s*.+',                  # 作者：张三
        r'收录于\s*.+',                # 收录于 专栏
        r'添加了问题',                  # xx 添加了问题
        r'添加了回答',                  # xx 添加了回答
        r'关注问题',                   # 关注问题
        r'写回答',                     # 写回答
        r'邀请回答',                   # 邀请回答
        r'查看全部',                   # 查看全部
        r'继续浏览',                   # 继续浏览
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 删除空行和多余空白
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # 过滤掉空行和短的无意义行
        if line and len(line) > 2:
            # 进一步过滤纯标点或短数字
            if not re.match(r'^[\d\s\.\,\!\?。，！？]*$', line):
                lines.append(line)
    
    # 合并连续的空格
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\s+', ' ', cleaned)  # 合并多个空格
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # 合并多个空行
    
    return cleaned.strip()


if __name__ == "__main__": 
    main()

