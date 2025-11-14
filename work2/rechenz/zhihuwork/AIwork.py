from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import csv
import json
import os

# 配置 options
options = Options()
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option('excludeSwitches', ['enable-automation'])
options.add_argument('--ignore-ssl-errors')
options.add_argument('--ignore-certificate-errors')
browser = webdriver.ChromiumEdge(options)
try:
    browser = webdriver.ChromiumEdge(options)
    wait = WebDriverWait(browser, 10)  # 设置显式等待

    browser.get('https://www.zhihu.com/topic/19554298/top-answers')

    # 检查 cookies 文件是否存在
    if os.path.exists("AIsolution/work2/zhihuwork/cookies.json"):
        with open("AIsolution/work2/zhihuwork/cookies.json", "r") as file:
            cookies = json.load(file)
            for cookie in cookies:
                browser.add_cookie(cookie)
        browser.refresh()

    with open("AIsolution/work2/zhihuwork/zhihu.csv", "w", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["问题名", "问题具体内容", "回答信息"])

        try:
            for num in range(2, 22):
                # 使用相对 XPath 和显式等待
                curelement = wait.until(
                    EC.presence_of_element_located((
                        By.XPATH,
                        f'//div[@id="TopicMain"]//div[{num}]//h2//a'
                    ))
                )

                browser.execute_script(
                    "arguments[0].scrollIntoView();", curelement)
                surl = curelement.get_attribute("href")

                if not surl:
                    print(f"无法获取第 {num} 个问题的链接")
                    continue

                browser.get(surl)

                results = []

                # 等待问题标题加载
                try:
                    problem_element = wait.until(
                        EC.presence_of_element_located((By.TAG_NAME, "h1"))
                    )
                    results.append(problem_element.text)
                except TimeoutException:
                    results.append("未获取到问题标题")

                # 等待并展开问题内容
                try:
                    button = wait.until(
                        EC.element_to_be_clickable(
                            (By.XPATH, '//*[contains(@class, "Button")]//text()="显示全部"/ancestor::button'))
                    )
                    browser.execute_script("arguments[0].click();", button)
                except TimeoutException:
                    print("未找到展开按钮或问题内容已完全显示")

                # 获取问题内容
                try:
                    content_element = wait.until(
                        EC.presence_of_element_located(
                            (By.CLASS_NAME, "RichText"))
                    )
                    results.append(content_element.text)
                except TimeoutException:
                    results.append("未获取到问题内容")

                # 获取回答内容
                for i in range(2, 12):
                    try:
                        answer_element = wait.until(
                            EC.presence_of_element_located((
                                By.XPATH,
                                f'//div[@data-za-detail-view-id="answer"]//div[@class="RichContent-inner"][{i-1}]'
                            ))
                        )
                        results.append(answer_element.text)
                    except TimeoutException:
                        results.append("未获取到回答内容")

                writer.writerow(results)
                browser.back()

        except Exception as e:
            print(f"处理过程中出现错误: {str(e)}")

except Exception as e:
    print(f"初始化或主流程出错: {str(e)}")
finally:
    browser.quit()

'''
煞笔东西
'''
