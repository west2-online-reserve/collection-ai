import csv
from pathlib import Path
from typing import Tuple

from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.webdriver import WebDriver
from bs4 import BeautifulSoup


ROOT_DIR: Path = Path(__file__).absolute().parent

TOPIC_URL: str = "https://www.zhihu.com/topic/19840977/hot"
SHOW_ALL_XPATH: str = (r"//*[@id='root']/div/main/div/div/div[1]/div[2]"
                        "/div/div[1]/div[1]/div[6]/div/div/div/button")

QUESTION_NUM: int = 12
ANSWER_NUM: int = 12


class Zhihu:
    def __init__(self) -> None:
        service: Service = Service()
        options: Options = Options()
        self.driver: WebDriver = WebDriver(options, service)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                    })"""})
        
        self.driver.get(TOPIC_URL)
        self.data = []

        #初始化文件
        self.file = open(ROOT_DIR/"data.csv",'w',encoding='utf-8',newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['问题','网址','问题描述','回答'])


    def login_qrcode(self) -> None:
        '''通过扫二维码登入'''
        print("Please login...")
        while True:
            try:
                login_page = self.driver.find_element(By.TAG_NAME, 'body')
                if login_page.get_attribute("style") == "overflow: auto;":
                    print("Login Successfully!")
                    break
            except:
                continue
    

    def get_answers(self, url) -> Tuple[str, list]:
        '''获取问题答案'''
        self.driver.get(url)
        try:  #点击显示全部按钮
            show_all = self.driver.find_element(By.XPATH, SHOW_ALL_XPATH)
            show_all.click()
        except Exception:
            pass
        finally:
            soup = BeautifulSoup(self.driver.page_source,'lxml')

        try: #获取问题的详细描述
            detail: str = soup.find("div", class_="QuestionRichText QuestionRichText--expandable").get_text()
        except AttributeError:
            try:
                detail: str = soup.find("div", class_="QuestionRichText QuestionRichText--collapsed").get_text()
            except AttributeError:
                detail: str = "No detailed description."

        #获取该问题的总答案数
        answer_count: int = int(soup.find("meta", attrs={"itemprop":"answerCount"}).get("content"))
        #不断刷新，以确保获取足够数量的答案
        tags = soup.find_all("div",class_ = "List-item", attrs={"tabindex": "0"})
        while len(tags) < ANSWER_NUM and len(tags) < answer_count:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.driver.execute_script("window.scrollTo(0, -10);")

            soup = BeautifulSoup(self.driver.page_source,'lxml')
            tags = soup.find_all("div",class_ = "List-item", attrs={"tabindex": "0"})

        answers: list = []
        for tag in tags:
            text_tag = tag.find('span', attrs={"itemprop":"text"})
            answers.append(text_tag.get_text())
        answers = answers[:ANSWER_NUM]

        return detail, answers


    def get_questions(self) -> None:
        soup = BeautifulSoup(self.driver.page_source, 'lxml')
        tags = soup.find_all('div', attrs={"itemprop":"zhihu:question"})

        #不断刷新，以确保获得足够数量的问题
        while len(tags) < QUESTION_NUM:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.driver.execute_script("window.scrollTo(0, -10);")

            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            tags = soup.find_all('div', attrs={"itemprop":"zhihu:question"})

        tags = tags[:QUESTION_NUM]
        for tag in tags:
            title: str = tag.get_text()
            url: str = tag.find("meta",attrs={"itemprop":"url"}).get("content")
            detail, answers = self.get_answers(url)
    
            data_unit = [title, url, detail] + answers
            self.writer.writerow(data_unit)


    def main(self) -> None:
        self.login_qrcode()
        self.get_questions()

        self.driver.quit()
        self.file.close()


if __name__ == "__main__":
    zhihu = Zhihu()
    zhihu.main()