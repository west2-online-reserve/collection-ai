from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from pathlib import Path
from collections import Counter
import time
from lxml import html
import csv
import random


def collect_questions(driver, wait, target_count=20, max_scrolls=10):
    seen_ids = set()
    questions = []
    for scroll_count in range(max_scrolls + 1):
        containers = driver.find_elements(
            By.CSS_SELECTOR, ".AnswerItem h2.ContentItem-title"
        )

        for container in containers:
            container_url = container.find_element(
                By.CSS_SELECTOR, "meta[itemprop='url']"
            ).get_attribute("content")
            container_title = container.find_element(By.CSS_SELECTOR, "a").text

            url_parts = container_url.split("/")
            container_question_id = url_parts[-1] if url_parts[-1] else url_parts[-2]
            if container_question_id not in seen_ids:
                seen_ids.add(container_question_id)
                questions.append(
                    {
                        "question_id": container_question_id,
                        "question_title": container_title,
                        "question_url": container_url,
                    }
                )

        if len(questions) >= target_count:
            break  # 已获取足够的问题，跳出循环

        if scroll_count == max_scrolls:
            print("已达到最大滚动次数，停止滚动")
            break

        old_count = len(containers)
        for _ in range(5):  # 向下滚动5次，每次滚动后等待0.5秒
            ActionChains(driver).scroll_by_amount(0, 800).perform()
            time.sleep(0.5)

        try:
            wait.until(
                lambda d: len(
                    d.find_elements(By.CSS_SELECTOR, ".AnswerItem h2.ContentItem-title")
                )
                > old_count
            )
        except TimeoutException:
            print("没有更多问题加载，停止滚动")
            break

    return questions[:target_count]  # 只保留前 target_count 个问题


def clean_answer(answer_item):
    answer_content_element = answer_item.find_element(
        By.CSS_SELECTOR, ".RichContent-inner span.RichText[itemprop='text']"
    )

    tree = html.fromstring(answer_content_element.get_attribute("outerHTML"))
    cards = tree.xpath(
        "//div[@class='RichText-ADLinkCardContainer' or @class='RichText-MCNLinkCardContainer']"
    )
    for card in cards:
        card.drop_tree()

    players = tree.xpath(".//video/../..")
    for player in players:
        player.drop_tree()

    parts = []

    # 遍历 tree 的直接子元素
    for child in tree:
        text = child.text_content().strip()
        if text:
            parts.append(text)

    # 用换行连接各部分
    draft_text = "\n".join(parts)

    return answer_content_element.text, draft_text


def crawl_question(driver, wait, question):
    time.sleep(2)  # 等待页面加载完成
    driver.get(question["question_url"])

    page_text = driver.find_element(By.TAG_NAME, "body").text
    if "40362" in page_text and "您当前请求存在异常" in page_text:
        raise RuntimeError("访问受限，可能需要登录或验证码验证")

    question_heading = wait.until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, ".QuestionHeader h1.QuestionHeader-title")
        )
    )  # 返回问题标题元素对象
    question_title = question_heading.text

    print("详情页标题：", question_title)

    question_descriptions = driver.find_elements(
        By.CSS_SELECTOR, ".QuestionHeader .QuestionRichText"
    )  # 返回问题描述元素对象列表
    print("问题描述元素数量：", len(question_descriptions))
    if not question_descriptions:
        question_content = ""
        print("问题内容为空")
    else:
        for attempt in range(1, 4):
            try:
                question_description = wait.until(
                    EC.visibility_of_element_located(
                        (By.CSS_SELECTOR, ".QuestionHeader .QuestionRichText")
                    )
                )
                print(question_description.get_attribute("class"))  # 输出问题描述的类名

                more_buttons = driver.find_elements(
                    By.CSS_SELECTOR, ".QuestionHeader button.QuestionRichText-more"
                )
                print("找到展开按钮数量：", len(more_buttons))
                if more_buttons:
                    more_button = wait.until(
                        EC.element_to_be_clickable(
                            (
                                By.CSS_SELECTOR,
                                ".QuestionHeader button.QuestionRichText-more",
                            )
                        )
                    )
                    more_button.click()

                question_content_element = wait.until(
                    EC.visibility_of_element_located(
                        (
                            By.CSS_SELECTOR,
                            ".QuestionHeader .QuestionRichText span.RichText[itemprop='text']",
                        )
                    )
                )
                question_content = question_content_element.text
                print("问题内容：", question_content)
                break  # 成功获取问题内容，跳出循环
            except TimeoutException:
                if attempt < 3:
                    time.sleep(1)  # 等待一秒后重试
                else:
                    raise TimeoutException("无法获取问题内容，已尝试3次")

    wait.until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.AnswerItem"))
    )  # 等待第一个回答元素对象出现

    seen_ids = set()
    answer_ids = []
    max_scrolls = 5
    misses = 0
    for scroll_count in range(max_scrolls + 1):
        answer_items = driver.find_elements(By.CSS_SELECTOR, "div.AnswerItem")
        for item in answer_items:
            item_id = item.get_attribute("name")
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                answer_ids.append(item_id)

        if len(answer_ids) >= 10:
            break  # 已获取足够的回答，跳出循环

        if scroll_count == max_scrolls:
            print("已达到最大滚动次数，停止滚动")
            break

        old_count = len(answer_items)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 2500);")
        for _ in range(5):  # 向下滚动5次，每次滚动后等待0.5秒
            ActionChains(driver).scroll_by_amount(0, 400).perform()
            time.sleep(0.5)

        try:
            wait.until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, "div.AnswerItem"))
                > old_count
            )
            misses = 0  # 重置 misses 计数器
        except TimeoutException:
            misses += 1
            print(f"第 {misses} 次未加载出新回答")
            if misses >= 2:
                print("没有更多回答加载，停止滚动")
                break
    answer_ids = answer_ids[:10]  # 只保留前10个回答ID
    print(len(answer_ids))
    print(answer_ids)

    answers = []
    for answer_id in answer_ids:
        answer_temp_item = driver.find_element(
            By.CSS_SELECTOR, f"div.AnswerItem[name='{answer_id}']"
        )
        answer_raw, answer_clean = clean_answer(answer_temp_item)
        answer_dict = {
            "answer_id": answer_id,
            "answer_content": answer_clean,
            "answer_content_raw": answer_raw,
        }
        answers.append(answer_dict)
        print(f"已处理回答ID：{answer_id}")
        print(len(answer_clean), "字符")
    print(len(answers), "个回答已处理")

    rows = []
    for answer in answers:
        row = {
            "question_id": question["question_id"],
            "question_title": question_title,
            "question_content": question_content,
            "answer_id": answer["answer_id"],
            "answer_content": answer["answer_content"],
        }
        rows.append(row)

    if len(rows) < 10:
        raise TimeoutException(f"回答不足：{len(rows)} 条")

    return rows


def main():
    options = webdriver.ChromeOptions()
    options.debugger_address = "127.0.0.1:9222"
    driver = webdriver.Chrome(options=options)
    try:
        driver.get("https://www.zhihu.com/")
        print("页面标题：", driver.title)

        wait = WebDriverWait(driver, 10)

        try:
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "button.AppHeader-profileEntry")
                )
            )
        except TimeoutException as error:
            input("登录状态未确认，请在浏览器中手动登录，完成后按回车：")
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "button.AppHeader-profileEntry")
                )
            )
        print("登录状态已确认")

        time.sleep(2)  # 等待页面加载完成
        driver.get("https://www.zhihu.com/topic/19554298/top-answers")

        questions = collect_questions(driver, wait, target_count=20, max_scrolls=10)
        print(f"已收集 {len(questions)} 个问题")
        for q in questions:
            print(q["question_id"], q["question_title"])

        fieldnames = [
            "question_id",
            "question_title",
            "question_content",
            "answer_id",
            "answer_content",
        ]
        csv_path = Path(__file__).parent / "data" / "zhihu_answers.csv"

        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                saved_rows = list(reader)
                saved_counts = Counter(row["question_id"] for row in saved_rows)
                saved_answer_ids = set(row["answer_id"] for row in saved_rows)
            print("已保存的行数：", len(saved_rows))
            print(
                "完整问题的个数：",
                sum(1 for count in saved_counts.values() if count >= 10),
            )
            for question_id, count in saved_counts.items():
                print(f"问题ID {question_id} 已保存 {count} 行")
        else:
            with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                saved_rows = []
                saved_counts = Counter()
                saved_answer_ids = set()

        all_rows = []
        batch = questions
        failures = []
        for index, question in enumerate(batch, start=1):
            if saved_counts[question["question_id"]] >= 10:
                print(f"第 {index} 题已完整，跳过")
                continue

            if index > 1:
                time.sleep(random.uniform(3, 6))  # 随机等待3-6秒

            print(
                f"第 {index}/{len(batch)} 题：{question['question_id']} {question['question_title']}"
            )

            try:
                rows = crawl_question(driver, wait, question)
                new_rows = [
                    row for row in rows if row["answer_id"] not in saved_answer_ids
                ]
            except TimeoutException as error:
                failures.append(
                    {
                        "question_id": question["question_id"],
                        "question_title": question["question_title"],
                        "reason": error.msg,
                    }
                )
                print("第", index, "题失败：", error.msg)
                continue

            all_rows.extend(new_rows)
            print(f"本题得到 {len(new_rows)} 个新回答")

            with open(csv_path, "a", encoding="utf-8", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(new_rows)
                print(f"已保存第 {index} 题 {len(new_rows)} 行")

        print(f"成功 {len(batch) - len(failures)} 题，失败 {len(failures)} 题")
        for failure in failures:
            print(
                "失败题目：",
                failure["question_id"],
                failure["question_title"],
                "原因：",
                failure["reason"],
            )

        with open(csv_path, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            read_rows = list(reader)
            print(len(read_rows), "行数据已从 CSV 文件读取")
        print("读回内容与写入一致：", saved_rows + all_rows == read_rows)  # True

        failures_path = Path(__file__).parent / "data" / "zhihu_failures.csv"
        failures_fieldnames = ["question_id", "question_title", "reason"]
        with open(failures_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=failures_fieldnames)
            writer.writeheader()
            writer.writerows(failures)
            print(f"已写入 {len(failures)} 条失败记录")

        question_id_set = set()
        answer_id_set = set()
        blank_num = 0
        for read_row in read_rows:
            question_id_set.add(read_row["question_id"])
            answer_id_set.add(read_row["answer_id"])
            if read_row["answer_content"].strip() == "":
                blank_num += 1
        print(len(question_id_set), "个不同问题ID")
        print(len(answer_id_set), "个不同回答ID")
        print(blank_num, "个回答内容为空")

        input()

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
