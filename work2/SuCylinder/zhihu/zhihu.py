from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep
import pathlib
import json
import random
import pandas
import re

edge_options = Options()
edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
edge_options.add_experimental_option("useAutomationExtension", False)
edge_options.add_argument("--disable-blink-features=AutomationControlled")
path = pathlib.Path(__file__).parent / "msedgedriver.exe"
service = EdgeService(executable_path=path)

edge = webdriver.Edge(service=service, options=edge_options)


# 获取问题
def get_questions(edge, url):
    edge.get(url)

    questions_elements = WebDriverWait(edge, 10).until(
        ec.presence_of_all_elements_located(
            (By.XPATH, '//div[@class = "QuestionItem-title"]/a')
        )
    )
    while len(questions_elements) < 20:
        sleep(1)
        # 滚动加载保证问题数大于20
        edge.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(1)
        edge.execute_script("window.scrollBy(0,-1000)")
        questions_elements = edge.find_elements(
            By.XPATH, '//div[@class = "QuestionItem-title"]/a'
        )

    questions = []

    for questions_element in questions_elements:
        question = {
            "question": questions_element.text,
            "answer_url": questions_element.get_attribute("href"),
        }
        questions.append(question)

    return questions[0:20]


# 获取回答
def get_answers(edge, url):
    edge.get(url)

    WebDriverWait(edge, 100).until(
        ec.presence_of_element_located((By.XPATH, '//span[@id = "content"]'))
    )

    # 滚动加载保证加载的回答大于10条
    while True:
        answer_elements = edge.find_elements(By.XPATH, '//span[@id = "content"]')
        if len(answer_elements) < 10:
            sleep(1)
            edge.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(1)
            edge.execute_script("window.scrollBy(0,-1000)")
        else:
            break

    ans = []

    # 获取回答的内容
    for answer_element in answer_elements[0:10]:
        answercontent = ""
        contents = answer_element.find_elements(By.XPATH, "./span/*")
        for content in contents:
            if len(content.text) > 0:
                answercontent += content.text
        answercontent = answercontent.strip().replace("\n", "").replace("\r", "")
        ans.append(answercontent)

    return ans


jsonfile = pathlib.Path(__file__).parent / "cookies.json"

with open(jsonfile, "r") as f:
    cookies = json.load(f)

edge.get("https://www.zhihu.com/")

for cookie in cookies:
    edge.add_cookie(cookie)
    print("add cookie")

edge.refresh()

if __name__ == "__main__":
    questions = get_questions(edge, "https://www.zhihu.com/topic/19556664/unanswered")
    print(len(questions))
    questions_and_answers = []
    for question in questions:
        sleep(random.uniform(0.3, 5.0))
        print(f"getting answers of {int(len(questions_and_answers)/10+1)} question")
        answers = get_answers(edge, question["answer_url"])

        if len(answers) > 0:
            for answer in answers:
                questions_and_answers.append(
                    {"question": question["question"], "answer": answer}
                )
        else:
            continue

    csv_file_path = pathlib.Path(__file__).parent / "zhihu.csv"
    df = pandas.DataFrame(questions_and_answers)
    df.to_csv(csv_file_path, index=False, encoding="utf-8")
