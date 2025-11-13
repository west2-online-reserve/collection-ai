import csv
import json
import os
import time
from typing import Dict, List

from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException


class ZhihuScraper:
    def __init__(self, cookie_file: str = "zhihu_cookies.json"):
        self.cookie_file = cookie_file
        self.driver = None
        self.wait = None

    # ---------- 初始化与登录 ----------

    def init_driver(self):
        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)

        self.driver = webdriver.Edge(options=options)
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
        })
        self.wait = WebDriverWait(self.driver, 12)

    def _logged_in(self) -> bool:
        try:
            self.driver.find_element(By.CSS_SELECTOR, "button.Button.AppHeader-login")
            return False
        except NoSuchElementException:
            return True

    def _load_cookies(self) -> bool:
        if not os.path.exists(self.cookie_file):
            return False

        with open(self.cookie_file, 'r', encoding='utf-8') as f:
            cookies = json.load(f)

        for cookie in cookies:
            cookie.pop('expiry', None)
            try:
                self.driver.add_cookie(cookie)
            except Exception:
                continue

        print("Cookies 已加载")
        return True

    def _save_cookies(self):
        cookies = self.driver.get_cookies()
        with open(self.cookie_file, 'w', encoding='utf-8') as f:
            json.dump(cookies, f)
        print("Cookies 已保存")

    def login(self):
        home = "https://www.zhihu.com/"
        self.driver.get(home)
        time.sleep(2)

        if self._load_cookies():
            self.driver.refresh()
            time.sleep(2)
            if self._logged_in():
                print("Cookie 登录成功")
                return
            print("Cookie 登录失效，请重新扫码")

        print("请扫码登录知乎，成功后按 Enter 继续...")
        input()
        self._save_cookies()

    # ---------- 基础工具 ----------

    @staticmethod
    def _clean_text(text: str) -> str:
        return ' '.join((text or '').replace('\u200b', ' ').split())

    def _scroll_down(self, step: int = 1600, pause: float = 1.0):
        self.driver.execute_script(f"window.scrollBy(0, {step});")
        time.sleep(pause)

    def _scroll_to_bottom(self, attempts: int = 3, pause: float = 1.0):
        for _ in range(max(attempts, 1)):
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            except Exception:
                break
            time.sleep(pause)

    # ---------- 话题列表 ----------

    def fetch_questions(self, topic_url: str, limit: int = 20) -> List[Dict[str, str]]:
        print(f"访问话题页面: {topic_url}")
        self.driver.get(topic_url)
        time.sleep(2)

        questions, seen = [], set()
        while len(questions) < limit:
            cards = self.driver.find_elements(By.CSS_SELECTOR, "h2.ContentItem-title a")
            for link in cards:
                title = link.text.strip()
                href = (link.get_attribute('href') or '').split('?')[0]
                if not title or not href or href in seen:
                    continue
                questions.append({'title': title, 'url': href})
                seen.add(href)
                print(f"已收集问题 {len(questions)}: {title[:30]}...")
                if len(questions) >= limit:
                    break
            if len(questions) >= limit:
                break
            self._scroll_down(step=3000, pause=1.2)
        return questions[:limit]

    # ---------- 问题页 ----------

    def _expand_question_detail(self):
        try:
            btn = self.driver.find_element(By.CSS_SELECTOR, "button.QuestionRichText-more")
            if btn.is_displayed():
                self.driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.4)
        except NoSuchElementException:
            pass

    def _is_full_answers_view(self) -> bool:
        try:
            current_url = self.driver.current_url or ""
        except Exception:
            return False
        return "/answers/" in current_url

    def _open_full_answers(self, question_url: str | None = None) -> bool:
        def answers_loaded() -> bool:
            return bool(self._collect_answer_nodes())

        if self._is_full_answers_view() and answers_loaded():
            return True

        # 先滚动到底部以曝光“查看全部回答”按钮
        self._scroll_to_bottom(attempts=2, pause=1.0)

        text_keywords = ["查看全部", "全部回答", "查看回答", "更多回答", "更多", "回答"]
        view_all_locators = [
            (By.XPATH, "//button[contains(normalize-space(),'查看全部') and contains(normalize-space(),'回答')]") ,
            (By.XPATH, "//a[contains(normalize-space(),'查看全部') and contains(normalize-space(),'回答')]") ,
            (By.XPATH, "//button[contains(normalize-space(),'全部回答')]") ,
            (By.XPATH, "//a[contains(normalize-space(),'全部回答')]") ,
            (By.XPATH, "//button[contains(@class,'QuestionMainAction') and contains(., '回答')]") ,
            (By.XPATH, "//a[contains(@class,'QuestionMainAction') and contains(., '回答')]")
        ]

        generic_locators = [
            (By.XPATH, "//button[contains(@class,'Button') and contains(., '回答')]") ,
            (By.XPATH, "//a[contains(@href, '/answers/')]") ,
            (By.CSS_SELECTOR, "div.QuestionMainAction button"),
            (By.CSS_SELECTOR, "div.QuestionMainAction a"),
            (By.CSS_SELECTOR, "button.QuestionAnswers-answerButton"),
            (By.CSS_SELECTOR, "button.Button--blue.Button--plain"),
            (By.CSS_SELECTOR, "button.Button--blue.Button--link"),
            (By.CSS_SELECTOR, "a[href*='answer-more']")
        ]

        def try_click(locators):
            for by, selector in locators:
                try:
                    elements = self.driver.find_elements(by, selector)
                except NoSuchElementException:
                    continue

                for element in elements:
                    if not element.is_displayed() or not element.is_enabled():
                        continue

                    text = (element.text or '').replace('\n', ' ').strip()
                    if text and not any(keyword in text for keyword in text_keywords):
                        continue

                    before_handles = tuple(self.driver.window_handles)
                    current_url = self.driver.current_url

                    try:
                        self.driver.execute_script("arguments[0].click();", element)
                    except Exception:
                        continue

                    time.sleep(1.0)

                    try:
                        self.wait.until(
                            lambda drv: bool(self._collect_answer_nodes())
                            or len(drv.window_handles) > len(before_handles)
                            or drv.current_url != current_url
                        )
                    except TimeoutException:
                        pass

                    self._switch_window_if_needed(before_handles)

                    if answers_loaded() or self._is_full_answers_view():
                        return True
            return False

        if not try_click(view_all_locators):
            try_click(generic_locators)

        if not answers_loaded():
            self._scroll_down(step=2000, pause=1.0)
            self._scroll_to_bottom(attempts=2, pause=1.0)

        if not (answers_loaded() or self._is_full_answers_view()) and question_url and "/answers/" not in (self.driver.current_url or ""):
            fallback = question_url.rstrip('/') + "/answers/created"
            if fallback != self.driver.current_url:
                try:
                    self.driver.get(fallback)
                    time.sleep(1.5)
                except Exception:
                    pass

        return self._is_full_answers_view() or answers_loaded()

    def _switch_window_if_needed(self, before_handles):
        try:
            after_handles = self.driver.window_handles
            if len(after_handles) > len(before_handles):
                new_handle = next(handle for handle in after_handles if handle not in before_handles)
                self.driver.switch_to.window(new_handle)
                time.sleep(0.5)
        except Exception:
            pass

    def _collect_answer_nodes(self):
        selectors = [
            "div.QuestionAnswers-answers div.List-item",
            "div.List-item[data-za-detail-view-element_name='AnswerItem']",
            "div.ContentItem.AnswerItem"
        ]
        for selector in selectors:
            nodes = self.driver.find_elements(By.CSS_SELECTOR, selector)
            if nodes:
                return nodes
        return []

    def _wait_for_new_answers(self, prev_count: int, timeout: float = 6.0, interval: float = 0.6) -> int:
        deadline = time.time() + timeout
        current = len(self._collect_answer_nodes())
        if current > prev_count:
            return current

        while time.time() < deadline:
            time.sleep(interval)
            current = len(self._collect_answer_nodes())
            if current > prev_count:
                break
        return current

    @staticmethod
    def _answer_id(node) -> str:
        attrs = [
            "data-za-detail-view-id",
            "data-za-extra-module",
            "data-answer-id",
            "id"
        ]
        for attr in attrs:
            try:
                value = node.get_attribute(attr) if node else None
            except StaleElementReferenceException:
                return ""
            if value:
                return value
        try:
            link = node.find_element(By.CSS_SELECTOR, "a[href*='/answer/']")
            href = link.get_attribute('href') or ''
            return href.split('/answer/')[-1].split('?')[0]
        except StaleElementReferenceException:
            return ""
        except Exception:
            return ""

    def _load_more_answers(self) -> bool:
        selectors = [
            "button.QuestionAnswers-AnswerBar-ExpandButton",
            "button.ContentItem-more"
        ]
        for selector in selectors:
            try:
                btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                if btn.is_displayed():
                    self.driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.8)
                    return True
            except NoSuchElementException:
                continue
        return False

    def fetch_question_detail(self, question: Dict[str, str], limit: int = 10) -> Dict[str, List[str]]:
        self.driver.get(question['url'])
        time.sleep(1.5)

        self._expand_question_detail()
        self._open_full_answers(question['url'])

        self._scroll_to_bottom(attempts=3, pause=1.0)
        self._wait_for_new_answers(0, timeout=10.0)

        try:
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.QuestionAnswers")))
        except TimeoutException:
            pass

        detail = "无问题描述"
        try:
            detail = self._clean_text(self.driver.find_element(By.CSS_SELECTOR, "div.QuestionRichText").text)
        except NoSuchElementException:
            pass

        answers, seen_ids, seen_text = [], set(), set()
        stuck_rounds = 0
        max_stuck_rounds = 8
        while len(answers) < limit:
            nodes = self._collect_answer_nodes()
            if not nodes:
                self._open_full_answers(question['url'])
                self._wait_for_new_answers(len(nodes), timeout=6.0)
                nodes = self._collect_answer_nodes()
                if not nodes:
                    break

            for node in nodes:
                if len(answers) >= limit:
                    break
                try:
                    try:
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", node)
                        time.sleep(0.1)
                    except Exception:
                        pass

                    try:
                        more_btn = node.find_element(By.CSS_SELECTOR, "button.ContentItem-more")
                        if more_btn.is_displayed() and more_btn.is_enabled():
                            self.driver.execute_script("arguments[0].click();", more_btn)
                            time.sleep(0.3)
                    except (NoSuchElementException, StaleElementReferenceException):
                        pass

                    try:
                        text_node = node.find_element(By.CSS_SELECTOR, "span.RichText")
                    except NoSuchElementException:
                        text_node = node
                    answer_text = self._clean_text(text_node.text)
                    answer_id = self._answer_id(node)
                except StaleElementReferenceException:
                    continue

                if not answer_text:
                    continue
                if answer_id and answer_id in seen_ids:
                    continue
                if not answer_id and answer_text in seen_text:
                    continue

                answers.append(answer_text)
                seen_text.add(answer_text)
                if answer_id:
                    seen_ids.add(answer_id)

            if len(answers) >= limit:
                break

            before = len(nodes)
            clicked = self._load_more_answers()
            if not clicked and before == len(nodes):
                self._scroll_down(step=2200, pause=1.2)
                self._scroll_to_bottom(attempts=2, pause=1.0)
            else:
                time.sleep(1.0)

            new_count = self._wait_for_new_answers(before, timeout=5.0)
            if new_count <= before:
                stuck_rounds += 1
                if stuck_rounds % 2 == 0:
                    self._scroll_to_bottom(attempts=2, pause=1.0)
                if stuck_rounds >= max_stuck_rounds:
                    break
            else:
                stuck_rounds = 0

        if len(answers) < limit:
            print(f"  实际仅抓取 {len(answers)} 条回答")

        return {
            'question': question['title'],
            'detail': detail,
            'answers': answers[:limit]
        }

    # ---------- 主流程 ----------

    def scrape_topic(self, topic_url: str, max_questions: int = 20, max_answers: int = 10):
        questions = self.fetch_questions(topic_url, max_questions)
        rows: List[Dict[str, str]] = []

        for idx, q in enumerate(questions, 1):
            print(f"\n处理问题 {idx}/{len(questions)}: {q['title']}")
            try:
                detail = self.fetch_question_detail(q, max_answers)
                for answer in detail['answers']:
                    rows.append({
                        '问题名': detail['question'],
                        '问题具体内容': detail['detail'],
                        '回答信息': answer
                    })
                print(f"完成问题 {idx}，回答数 {len(detail['answers'])}")
            except Exception as err:
                print(f"处理问题失败: {err}")
        return rows

    # ---------- 保存 ----------

    @staticmethod
    def save_to_csv(rows: List[Dict[str, str]], filename: str = 'zhihu_data.csv'):
        if not rows:
            print("无数据可保存")
            return
        with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['问题名', '问题具体内容', '回答信息'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"数据已保存至 {filename}，共 {len(rows)} 行")

    # ---------- 启动 ----------

    def run(self, topic_url: str, max_questions: int = 20, max_answers: int = 10):
        try:
            self.init_driver()
            self.login()
            rows = self.scrape_topic(topic_url, max_questions, max_answers)
            self.save_to_csv(rows)
        finally:
            if self.driver:
                self.driver.quit()


if __name__ == '__main__':
    topic = "https://www.zhihu.com/topic/19554298/top-answers"
    ZhihuScraper().run(topic, max_questions=20, max_answers=10)

