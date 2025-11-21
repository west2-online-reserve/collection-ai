from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import csv
import random

class ZhihuSpider:
    def __init__(self):
        # 设置浏览器选项 - 不自动关闭
        self.option = webdriver.ChromeOptions()
        self.option.add_experimental_option("detach", True)
        # 添加反检测选项
        self.option.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.option.add_argument('--disable-blink-features=AutomationControlled')
        
        # 创建浏览器驱动
        self.driver = webdriver.Chrome(options=self.option)
        self.data = []
        
    def random_delay(self, min_sec=1, max_sec=3):
        """随机延迟，模拟人类行为"""
        time.sleep(random.uniform(min_sec, max_sec))
    
    def login(self):
        """手动登录"""
        print("正在打开知乎...")
        self.driver.get('https://www.zhihu.com/signin')
        print("请手动登录知乎，登录完成后按回车...")
        input()
        print("登录完成！")
        return True
    
    def crawl_hot_topic(self):
        topic_url = "http://zhihu.com/topic/19556784/hot"
        self.driver.get(topic_url)
        time.sleep(3)

        # 使用 Selenium 获取问题元素
        question_elements = self.driver.find_elements(By.CSS_SELECTOR, '[itemprop="zhihu:question"]')
        
        # 不断滚动，直到获取10个问题
        while len(question_elements) < 10:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            question_elements = self.driver.find_elements(By.CSS_SELECTOR, '[itemprop="zhihu:question"]')
            print(f"已找到 {len(question_elements)} 个问题")
        
        # 提取问题信息
        questions = []
        for element in question_elements[:10]:  # 只取前10个问题
            try:
                # 获取问题标题和链接
                title_element = element.find_element(By.CSS_SELECTOR, '[itemprop="name"]')
                url_element = element.find_element(By.CSS_SELECTOR, '[itemprop="url"]')
                
                title = title_element.get_attribute('content')
                url = url_element.get_attribute('content')
                
                if title and url:
                    questions.append({
                        "title": title,
                        "url": url
                    })
                    print(f"找到问题: {title}")
            except Exception as e:
                continue
        
        # 爬取每个问题的回答
        for i, question in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] 正在爬取问题: {question['title']}")
            
            # 随机延迟，避免请求过快
            self.random_delay(2, 4)
            
            # 在新标签页打开问题
            self.driver.execute_script(f"window.open('{question['url']}');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            time.sleep(3)

            try:
                show_all = self.driver.find_element(By.XPATH, "//button[contains(text(),'显示全部')]")
                show_all.click()
                time.sleep(2)
            except:
                pass
            
            # 获取问题详情
            question_detail = "无详细描述"
            try:
                detail_element = self.driver.find_element(By.CSS_SELECTOR, ".QuestionRichText")
                question_detail = detail_element.text.strip()[:300]
            except:
                pass

            answers = []
            for scroll in range(15):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                answer_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.List-item[tabindex='0']")
                for element in answer_elements:
                    try:
                        text_element = element.find_element(By.CSS_SELECTOR, 'span[itemprop="text"]')
                        text = text_element.text.strip()
                        if text and len(text) > 50 and text not in answers:
                            answers.append(text)
                            if len(answers) >= 20:
                                break
                    except:
                        continue                
                if len(answers) >= 20:
                    break    
            # 保存数据
            for answer in answers:
                self.data.append({
                    "question_title": question['title'],
                    "question_detail": question_detail,
                    "answer_content": answer
                })

            # 关闭当前标签页，回到主页面
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])
            self.random_delay(1, 2)
    
    def save_to_csv(self, filename="zhihu_data.csv"):
        """保存数据到CSV"""
        if not self.data:
            print("没有数据可保存")
            return
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["question_title", "question_detail", "answer_content"])
            writer.writeheader()
            writer.writerows(self.data)
    
    def run(self):
        """运行爬虫"""
        try:
            if not self.login():
                return
            self.crawl_hot_topic()

            self.save_to_csv()
            print("文件已保存为: zhihu_10q_20a.csv")
            
        except Exception as e:
            print(f"程序运行出错: {e}")

# 运行爬虫
if __name__ == "__main__":
    spider = ZhihuSpider()
    spider.run()