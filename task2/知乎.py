import pandas as pd
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


def init():
    """初始化浏览器"""
    service = Service('./py/task2/chromedriver-win64/chromedriver.exe')  # chromedriver路径
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def fetch_cntt(driver):
    """通过浏览器打开页面并返回内容"""
    driver.get('https://www.zhihu.com/topic/19554298/top-answers')
    time.sleep(20)
    return driver.page_source


def ext_que(driver, start=2, end=52):
    """提取问题信息"""
    questions = []
    index=start
#a_elements = driver.find_elements(By.XPATH, '//a[@target="_blank"  and @data-za-detail-view-element_name="title"]')
    for index in range(start, end+1):
        
        xpath = f"/html/body/div[1]/div/main/div/div[1]/div[2]/div[4]/div/div/div/div[{index}]/div/div/h2/div/a"
        try:
            element = driver.find_element(By.XPATH, xpath)
            #element = a_elements[index-2]
            url1 = element.get_attribute('href')
            title = element.text.strip()
            questions.append({'url': url1, 'title': title})
            print(f"问题 {index-1} 提取成功\ntitle: {title}\nurl: {url1}")
            time.sleep(3)
            if index%3==0:
                driver.execute_script("window.scrollBy(0, 1500);")
                time.sleep(2)
        except Exception as e:
            print(f"问题 {index-1} 提取失败，尝试滚动页面：{e}")
            driver.execute_script("window.scrollBy(0, 1500);")
            time.sleep(3)
    return questions


def _ext_aser(driver, question):
    """获取特定问题的回答"""
    url = question['url']
    driver.get(url)
    time.sleep(2)
    answers = []

    # 尝试点击 "查看全部" 按钮
    try:
        div_element = driver.find_element(By.XPATH, "//div[contains(@class, 'Card') and contains(@class, 'ViewAll')]")
        view_all_button = div_element.find_element(By.XPATH, ".//a[contains(text(), '查看全部')]")
        view_all_button.click()
        time.sleep(2)
    except Exception as e:
        print(f"未找到 '查看全部' 按钮：{e}")

    # 提取答案
    for i in range(2, 22):
        xpath = f"/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div/div/div/div/div[2]/div/div[{i}]/div/div/div[2]/span[1]/div/div/span"
        try:
            element = driver.find_element(By.XPATH, xpath)
            paragraphs = element.find_elements(By.TAG_NAME, 'p')
            answer = ''
            for p in paragraphs:
                answer += p.text.strip() + '\n'
            answers.append(answer)
            print(f"第 {i-1} 个回答提取成功")
            driver.execute_script("window.scrollBy(0, 3000);")
            time.sleep(5)
        except Exception as e:
            print(f"第 {i-1} 个回答提取失败：{e}")
            driver.execute_script("window.scrollBy(0, 5000);")
            time.sleep(5)
    return answers


if __name__ == '__main__':
    driver = init()
    fetch_cntt(driver)
    # 获取问题和URL
    questions = ext_que(driver)
    data = []                                                            
    # 获取每个问题的答案
    for question in questions:
        answer = _ext_aser(driver, question)
        answers = "\n".join(answer)  # 将所有回答合并成一个字符串，并使用换行符分隔
        data.append({
            'question': question['title'],
            'answers': answers
        })
    # 数据整理并保存到 CSV
    df = pd.DataFrame(data)
    df.to_csv(r'D:\code\py\task2\zhihu_data.csv', index=False, encoding='utf-8-sig')
    print("数据已保存到 'zhihu_data.csv'")
    driver.quit()
