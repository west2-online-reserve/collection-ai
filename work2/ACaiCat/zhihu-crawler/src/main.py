import json
import re
import time
from typing import Optional, Any

import pandas
from bs4 import BeautifulSoup, ResultSet, Tag
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.webdriver import WebDriver

from constants import ANSWER_COUNT
from constants import QUESTION_COUNT
from question import Question


def main():
    service: Service = Service()
    options: Options = Options()
    driver: WebDriver = WebDriver(options, service)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
        Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
        })"""})

    driver.get("https://www.zhihu.com/topic/28811749/hot")

    print("Please log in on the page of driver...")

    # wait for login
    while True:
        try:
            driver.find_element(By.XPATH, r'//*[@id="TopicMain"]/div[4]/div/div/div/div[5]')
            break
        except NoSuchElementException:
            continue

    question_dict: dict[str, Question] = {}

    while len(question_dict) < QUESTION_COUNT:
        question_dict.clear()
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        topic_soup: BeautifulSoup = BeautifulSoup(driver.page_source, "html.parser")
        question_tags: ResultSet[Tag] = topic_soup.find_all("div", class_="List-item TopicFeedItem")

        for tag in question_tags:
            answer_tag: Optional[Tag] = tag.find("div", class_="ContentItem AnswerItem")
            if not answer_tag:
                continue

            title_ref: Tag = tag.find("a", {"data-za-detail-view-element_name": "Title"})

            answer_data: dict[str, Any] = json.loads(answer_tag.get("data-zop"))
            title: str = answer_data["title"]

            link: str = title_ref.get("href")
            link = re.sub(r"/answer/\d*", "", link)

            question_dict[title] = Question(
                title=title,
                url=f"https:{link}",
                body="",
                answers=[]

            )

    questions = list(question_dict.values())[:QUESTION_COUNT]

    for question in questions:
        driver.get(question.url)
        time.sleep(1)
        retry = 0
        while len(question.answers) < ANSWER_COUNT:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.execute_script("window.scrollTo(0, -10);")
            question.answers.clear()

            # noinspection PyBroadException
            try:
                button = driver.find_element(By.CSS_SELECTOR, ".QuestionMainAction.ViewAll-QuestionMainAction")
                button.click()

            except Exception:
                pass

            try:
                button = driver.find_element(By.XPATH,
                                             r'//*[@id="root"]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/button')
                button.click()

            except Exception:
                pass

            question_soup: BeautifulSoup = BeautifulSoup(driver.page_source, "html.parser")
            body: Optional[Tag] = question_soup.find("div", class_="QuestionRichText QuestionRichText--expandable")
            if body is not None:
                question.body = body.get_text()

            answer_tags: ResultSet[Tag] = question_soup.find_all("div", class_="RichContent RichContent--unescapable")
            question.answers = [tag.get_text() for tag in answer_tags]

            retry += 1
            if retry >= 100:
                break

        question.answers = question.answers[:ANSWER_COUNT]

    question_data: list[dict[str, Any]] = []
    for question in questions:
        row: dict[str, Any] = {
            "question": question.title,
            "body": question.body,
            "answers": question.answers,
            "url": question.url,
        }
        question_data.append(row)
    question_frame: pandas.DataFrame = pandas.DataFrame(question_data)
    question_frame.to_csv("question.csv", index=False, encoding='utf-8-sig')
    print(f"Exported {len(question_frame)} records...")


if __name__ == '__main__':
    main()
