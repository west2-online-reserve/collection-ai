from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService #edge驱动
from selenium.webdriver.support.ui import WebDriverWait  #等待元素
from selenium.webdriver.common.by import By  #路径
from selenium.webdriver.support import expected_conditions as EC  #webdriverwait中需要的参数
from selenium.webdriver.common.action_chains import ActionChains  #动作链
from selenium.webdriver.edge.options import Options  #一些设置，避免被识别为自动化脚本
import json
import time
import random
import csv



class Zhihu_Crawler():
    def __init__(self):
        self.question_num = 25
        self.answer_num = 10
        self.question_list = []
        self.questions_window = None
        #浏览器初始化
        self.service = EdgeService(r"C:\Program Files (x86)\Microsoft\Edge\Application\edgedriver_win64 (1)\msedgedriver.exe")
        self.options = Options()
        
        # 常见的反检测选项
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)

        # 其他有用的常规选项，让浏览器更像普通用户
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        # options.add_argument("--headless") # 无头模式更容易被检测，非必要勿用
        
        # 非常重要：设置一个常见的用户代理
        self.options.add_argument(
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0'
                             )
        
        self.driver = webdriver.Edge(service = self.service,options=self.options)

        #初始化csv文件
        self.fieldname = ['问题','详细问题','回答表']
        with open('content2.csv','w',encoding= 'utf-8-sig')as f:
            writer = csv.DictWriter(f,fieldnames=self.fieldname)
            writer.writeheader()

    #初次登入，获取cookies
    def first_login(self,login_url):
        service = EdgeService(r"C:\Program Files (x86)\Microsoft\Edge\Application\edgedriver_win64 (1)\msedgedriver.exe")
        self.driver = webdriver.Edge(service = service)
        self.driver.get(login_url)
        input('按任意键继续')
        time.sleep(3)
        cookies = self.driver.get_cookies()
        with open('cookies.json','w') as f:
            json.dump(cookies,f,ensure_ascii=False)
        print(cookies)
        


    #打开Edge浏览器

    def login_by_cookies(self,url):
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        time.sleep(random.uniform(1,2))
        #driver.maximize_window()

        
        self.driver.get(url)
        time.sleep(random.uniform(1,2))
        with open('cookies.json','rb') as f:
            cookies = json.load(f)
        for cookie in cookies:
            self.driver.add_cookie(cookie)
        time.sleep(random.uniform(1,2))
        self.driver.refresh()
        time.sleep(random.uniform(1,2))

    def get_hot_topics(self):

        wait = WebDriverWait(self.driver, 10)
        element = wait.until(
        EC.presence_of_element_located((By.XPATH,'//*[@id="root"]/div/div[3]/header/div/div[1]/div[1]/nav/a[3]')
        ))
        
        time.sleep(random.uniform(1, 2))
        #点击
        element.click()
        #self.driver.execute_script("window.scrollBy(0,300);")
        element1 = wait.until(
            EC.presence_of_all_elements_located(
                (By.XPATH,'//*[@id="TopstoryContent"]/div/div[2]/div[1]//section/div[2]/a')
            )
        )
        
        print('=========')
        print(element1)
        
        time.sleep(random.uniform(1,3))
        return element1
    
    def enter_to_topic(self):
        wait = WebDriverWait(self.driver,10)
        elements = wait.until(
            EC.presence_of_all_elements_located(
                (By.XPATH,'/html/body/div[3]/div[1]/div/div/div[2]/div//div/div/a[1]')
            )
        )
        
        time.sleep(random.uniform(1,2))
        elements[0].click()
        
    def get_qustion_list(self):
        #切换到最新窗口
        windows = self.driver.window_handles
        self.driver.switch_to.window(windows[-1])

        self.questions_window = self.driver.current_window_handle

        actions = ActionChains(self.driver)
        wait = WebDriverWait(self.driver,10)
        while len(self.question_list) <= self.question_num +5:
            actions.scroll_by_amount(0,1000).perform()
            
            self.question_list = wait.until(
                EC.presence_of_all_elements_located(
                    (By.XPATH,'//*[@id="TopicMain"]/div[4]/div[2]/div/div//div/div/div/h2/div/a')
                )
            )
            print(len(self.question_list))
            time.sleep(random.uniform(1,2))
        print(len(self.question_list))

    def get_answer(self):
        
        current_question_num = 0
        while current_question_num < self.question_num:
            self.driver.switch_to.window(self.questions_window)

            self.driver.execute_script("arguments[0].click();", self.question_list[current_question_num])
            
            #切换到最新窗口
            windows = self.driver.window_handles
            self.driver.switch_to.window(windows[-1])
            try:
                wait = WebDriverWait(self.driver,10)
                element = wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH,'//*[@id="root"]/div/main/div/div/div[3]/div[1]/div/div[1]/a')
                    )
                )
                current_question_num += 1
                time.sleep(random.uniform(1,2))
                element.click()
            except:
                continue

            question_text = ''
            question_content_text = ''
            try:
                wait = WebDriverWait(self.driver,10)
                question = wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH,'//*[@id="root"]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/h1')
                        )   
                    )
                print(question)
                time.sleep(1)
                question_text = question.text
                #print(question.text)
                #print(question_text)
                time.sleep(1)
            except:
                print('无问题')
            try:
                question_content = self.driver.find_element(By.XPATH,'//*[@id="root"]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/span')
                question_content_text = question_content.text
            except Exception as e:
                print("无详细问题")
            try:
                elements = []
                while len(elements)<=self.answer_num:
                    actions = ActionChains(self.driver)
                    actions.scroll_by_amount(0,-200).scroll_by_amount(0,5000).perform()
                    wait = WebDriverWait(self.driver,20)
                    elements = wait.until(
                        EC.presence_of_all_elements_located(
                            (By.XPATH,'//*[@id="QuestionAnswers-answers"]//div[@class = "List-item"]')
                            )
                        )
                    print(len(elements))
                    time.sleep(1)
                elements_content = [i.text for i in elements]
                dic = {'问题':question_text,'详细问题':question_content_text,'回答表':elements_content}
                with open('content2.csv','a',encoding= 'utf-8-sig') as file:
                        writer = csv.DictWriter(file,fieldnames=self.fieldname)
                        writer.writerow(dic)
                print(len(elements))
                print(elements_content[0])
            except Exception as e:
                print(e)
            self.driver.close()
            

login_url = 'https://www.zhihu.com/signin'
url = 'https://www.zhihu.com/topics'


#first_login(login_url)
#driver = login_by_cookies(url)
#elements = get_hot_topics(driver)
crawler = Zhihu_Crawler()
crawler.driver.maximize_window()
crawler.login_by_cookies(url)
crawler.enter_to_topic()
crawler.get_qustion_list()
crawler.get_answer()
time.sleep(100)