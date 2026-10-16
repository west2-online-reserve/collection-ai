import csv
import json
import random
import time

from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

COOKIE_PATH = (
    r"D:\Files\Code\Python\collection-ai\foundation\hao888myt\task2\zhi_hu\cookies.json"
)

if __name__ == "__main__":
    # 已劳累，针对反爬的伪装
    # 我被知乎针对了...哭
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Edge(options=options)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
        },
    )

    driver.get("https://www.zhihu.com")

    with open(COOKIE_PATH, "r", encoding="utf-8") as f:
        cookies = json.load(f)

    for cookie in cookies:
        driver.add_cookie(cookie)  # type: ignore
    time.sleep(3)

    driver.get("https://www.zhihu.com/topic/19845595/top-answers")
    time.sleep(3)

    for i in range(2):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # type: ignore
        time.sleep(2)

    urls: list[str] = []

    links = driver.find_elements(
        By.CSS_SELECTOR, 'a[data-za-detail-view-element_name="Title"]'
    )

    for link in links:
        url = link.get_attribute("href").split("/answer/")[0]  # type: ignore
        if url not in urls:
            urls.append(url)

    urls = urls[:20]

    results = []

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        driver.get(url)
        time.sleep(3)

        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1, 4))

            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(random.uniform(1, 2))

            driver.execute_script("window.scrollBy(0, -300);")
            time.sleep(random.uniform(1, 2))

            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(random.uniform(1, 3))

        try:
            question = driver.find_element(
                By.CLASS_NAME, "QuestionHeader-title"
            ).get_attribute("textContent")
            question = question.strip() if question else ""
        except NoSuchElementException:
            question = ""

        try:
            question_detail = driver.find_element(
                By.CSS_SELECTOR, ".QuestionRichText"
            ).get_attribute("textContent")
            question_detail = question_detail.strip() if question_detail else ""
        except NoSuchElementException:
            question_detail = ""

        answers = driver.find_elements(By.CSS_SELECTOR, ".RichContent-inner")[:20]

        count = 0
        for a in answers:
            text = a.text.strip()
            if text:
                text = " ".join(text.split())
                results.append([question, question_detail, text])
                count += 1

        print(f"爬到{count}条回答")

    with open("zhihu_answers.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["问题", "问题详情", "回答"])
        writer.writerows(results)
