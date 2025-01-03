'''用于爬取知乎的爬虫 爬取速度过慢 目前改进想法 把网页存下来 用其他库来解析'''
from selenium import webdriver
from selenium.webdriver.common.by import By
from lxml import etree
from time import sleep
import csv

class ZhiHuSpider(object):
    """用于爬取知乎的类"""
    def __init__(self) -> None:
        #隐藏日志并初始化
        self.options = webdriver.EdgeOptions()
        self.options.add_experimental_option('excludeSwitches',['enable-logging'])
        self.options.add_argument("--disable-blink-features=AutomationControlled")#避免检测的option代码
        self.options.add_argument('--disable-gpu')  # 如果使用GPU加速，需要禁用
        #self.options.add_argument('--headless')  # 无头模式
         # 添加用户代理伪装
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
        self.options.add_argument(f"user-agent={user_agent}")
        self.wd = webdriver.Edge(options=self.options)
        self.target = 'https://www.zhihu.com/topic/19554298/top-answers'
        self.titles = []
        self.urls = []
        self.info = []
          
        
    def login(self):
        """登录qq的函数"""
        self.wd.get('https://graph.qq.com/oauth2.0/show?which=Login&display=pc&client_id=100490701&redirect_uri=https%3A%2F%2Fwww.zhihu.com%2Foauth%2Fcallback%2Fqqconn%3Faction%3Dlogin%26from%3D&response_type=code&scope=get_info%2Cget_user_info%2Cget_other_info%2Cadd_t%2Cadd_pic_t%2Cget_fanslist%2Cget_idollist%2Cadd_idol%2Cadd_share')
        sleep(3)
        self.wd.switch_to.frame("ptlogin_iframe")
        buttom = self.wd.find_element(By.XPATH,'//*[@id="qlogin_list"]/a')
        buttom.click()
        sleep(3)
    
    def get_urls(self):
        """获取问题的问题名和链接"""
        self.wd.get(f'{self.target}')
        page = 5
        sleep(3)
        #滚动页面
        for _ in range(page):
            self.wd.execute_script('window.scrollTo(0,document.body.scrollHeight)')
            sleep(3)
        #sleep(2)
        page_text = self.wd.page_source
        tree = etree.HTML(page_text)
        a_list = tree.xpath('//*[@id="TopicMain"]//a[@data-za-detail-view-element_name="Title"]')
        for a in a_list:
            title = a.xpath('./text()')
            href = a.xpath('./@href')
            if title[0] in self.titles:
                continue
            else:
                self.titles.append(title[0])
                self.urls.append('https:'+href[0])
            print(title[0],'https:'+href[0])
        print(f'共爬取了{len(self.titles)}条问题！')
        

    def save_info(self):
        """把爬取的内容存入csv文件中"""
        with open('zhihu.csv', 'a', newline='', encoding='utf-8') as csvfile:
            # 创建 CSV 写入器
            csvwriter = csv.writer(csvfile)
            # 仅在文件为空时写入表头
            if csvfile.tell() == 0:
                csvwriter.writerow(['标题','链接'])
            for title,link in zip(self.titles,self.urls):
                csvwriter.writerow([title,link])
        print('文件存储成功！')

    def full_url(self,url):
        """打开页面展开回答，下滑页面"""
        #携带伪装的cookie
        new_cookie = {'name': '__zse_ck', 'value': '003_bqm+1Ra4xCO4peJ8SdE6/oi7Ub1v=nWPY8h4DIPddPQKF=7CdxKWUb4Muth9pLBmiJcjq3CBQGBGt1iaBEyFOv9B8s7890gPAL2hiE9KeR2O'}  
        self.wd.delete_cookie('__zse_ck')  
        self.wd.add_cookie(new_cookie)
        self.wd.get(url)
        sleep(3)
        try:
            button = self.wd.find_element(By.XPATH,'//*[@id="root"]/div/main/div/div/div[3]/div[1]/div/div[1]/a')
            button.click()
        except Exception:
            print("没有找到'展开全部回答'按钮或已加载完毕。")
        sleep(3)
        try:
            button1 = self.wd.find_element(By.XPATH,'//div[@class="css-eew49z"]//button')
            if button1:button1.click()
            sleep(3)
        except Exception:
            print("没有找到'展开全部描述'按钮或已加载完毕。")
        for _ in range(10):
            self.wd.execute_script('window.scrollTo(0,-10)')
            self.wd.execute_script('window.scrollTo(0,document.body.scrollHeight)')
            sleep(2)
        
    def save_data(self):
        with open('./知乎回答.csv', 'a', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                    writer.writerow(['问题', '问题描述', '作者', '回答'])
            for i in self.info:
                writer.writerow(i)
            self.info = []

    def spider_one_url(self,url):
        """爬取页面中的问题描述寄回答"""
        self.full_url(url)
        page_text = self.wd.page_source
        tree = etree.HTML(page_text)
        try:
            title = tree.xpath('//*[@id="root"]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/h1/text()')[0]
            describption = tree.xpath('//div[@class="css-eew49z"]//text()')#问题描述
            division_list = tree.xpath('//div[@class="List-item"]')#不同回答区域
            #print(describption)
            for div in division_list:
                author = div.xpath('.//div[@class="AuthorInfo-content"]//span[@class="UserLink AuthorInfo-name"]//text()')
                text =''.join(div.xpath('.//div[@class="RichContent-inner"]//p/text()'))
                self.info.append([title,describption,author[0],text])
                #print(author[0])
                #print(text)
            self.save_data()
        except Exception:
            return 
        
    def go(self):
        """启动！"""
        self.login()
        self.get_urls()
        self.save_info()
        #self.spider_one_url('https://www.zhihu.com/question/328444462/answer/1194782618')
        for web in self.urls:
           self.spider_one_url(web)
        
        

if __name__=='__main__':
    s = ZhiHuSpider()
    s.go()

    