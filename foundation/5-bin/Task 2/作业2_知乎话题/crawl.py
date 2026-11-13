from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json
from pathlib import Path


def main():
    driver = webdriver.Chrome()
    try:
        driver.get("https://www.zhihu.com/")
        print("页面标题：", driver.title)

        cookie_path = Path(__file__).parent / "private" / "cookies.json"
        json_text = cookie_path.read_text(encoding="utf-8")
        cookies = json.loads(json_text)

        # for cookie in cookies:
        #     driver.add_cookie(cookie)

        driver.refresh()  # 刷新页面以应用cookie

        wait = WebDriverWait(driver, 10)

        try:
            wait.until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, "button.AppHeader-profileEntry")
                )
            )
        except TimeoutException as error:
            input("登录状态未确认，请在浏览器中手动登录，完成后按回车：")
            wait.until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, "button.AppHeader-profileEntry")
                )
            )
        print("登录状态已确认")

        # cookies = driver.get_cookies()
        # json_text = json.dumps(cookies, ensure_ascii=False, indent=2)
        # cookie_path.write_text(json_text, encoding="utf-8")
        # print("Cookie 已更新")

        driver.get("https://www.zhihu.com/topic/19554298/top-answers")

        question_meta = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "h2.ContentItem-title meta[itemprop='url']")
            )
        )
        question_url = question_meta.get_attribute("content")
        print(question_url)

        parts = question_url.split("/")
        question_id = parts[-1] if parts[-1] else parts[-2]
        print(question_id)

        question_link = wait.until(
            EC.visibility_of_element_located(
                (By.CSS_SELECTOR, "h2.ContentItem-title a")
            )
        )
        question_title = question_link.text
        print(question_title)

        driver.get(question_url)

        json_dict = {
            "question_id": question_id,
            "question_title": question_title,
            "question_url": question_url,
        }
        json_text = json.dumps(json_dict, ensure_ascii=False, indent=2)
        json_path = Path(__file__).parent / "data" / "sample_question.json"
        json_path.write_text(json_text, encoding="utf-8")

        input()

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
