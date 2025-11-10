from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By #导入元素定位模块
from selenium.webdriver.support.ui import WebDriverWait  #引入显示等待
from selenium.webdriver.support import expected_conditions as ec   #引入显示等待结束需要的情况
import time
import csv
import random
from pprint import pprint

def get_page():  #定义创建网页的函数
    arg = Options()  #创建设置浏览器可选参数对象
    #arg.add_argument(" --no-sandbox")
    arg.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67")
    arg.add_experimental_option("excludeSwitches", ["enable-automation"])
    arg.add_experimental_option("useAutomationExtension", False)
    arg.add_experimental_option("detach",True) #保持浏览器打开状态，默认是代码运行完就关闭，experimental_option()方法设置实验性参数，第一个参数是默认分离表示浏览器网页和代码进程分离，第二个参数是是否保持打开
    page = webdriver.Edge(service = Service(r'D:\建模学习listing\msedgedriver.exe'),options = arg) #创建浏览器对象，指定浏览器驱动路径，并附上先前创建的参数对象
    page.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })                  #AI抄来的,防止被看出来是自动化浏览器的参数设置
        """
    })
    return page
def switch_page():   #切换网页函数
    global p1
    p_nums = p1.window_handles #获取在原网页基础上打开的所有窗口句柄，返回列表
    p1.switch_to.window(p_nums[-1]) #从p1切换到输入的句柄对应网页的window，一般是到最后一个所以是-1
    p1.implicitly_wait(4)
def scroll_page(max_scrolls):  #输入最多滚动次数，十条评论其实一次就差不多了就差不多
    global p1
    scroll_count = 0   #下滑函数，其实在这里没啥用，做笔记作用
    while scroll_count < max_scrolls: #设置最大下滑次数为限制
        p1.execute_script("window.scrollTo(0, document.body.scrollHeight);")  #用js滚动页面，到最低处
        time.sleep(2)  #等待数据加载完再滚动
        scroll_count += 1
if __name__ == "__main__":
    p1 = get_page()
    p1.get('https://www.zhihu.com/')

    cookies = [{
        'name':'z_c0',
        'value':'2|1:0|10:1761193880|4:z_c0|92:Mi4xZHJ2dGpnQUFBQUF6NTlNMXppaEFHeVlBQUFCZ0FsVk5sdjNtYVFDSjVnQnBBUjZXSThwc0MyUV9FeDF3dEJPQzB3|47e10a4d1912e6bdda8df7690cfba2f65ea082876d3d3e13d34d32971aed1624',
        'domain': '.zhihu.com'
        },
    {
        'name':'captcha_session_v2',
        'value':'2|1:0|10:1761193854|18:captcha_session_v2|88:cWZ5T0xWaCtNWnQ3bENkMlpFNkZtR0VEeUtTL0pEZWk1UU5KVGpid3FNRHhJNFE1M1Q5RGkvVFd5VExJRXlmMw==|1e241bc29a28e09d6f17a175555bd1e1ccd368acb6e4f825530427e552cba147',
        'domain': '.zhihu.com'
        },
    {
        'name':'_xsrf',
        'value':'ef85370a-f961-4349-a6e5-2ff58a98b36f',
        'domain': '.zhihu.com'
        },
    {
        'name':'DATE',
        'value':'1760942134182',
        'domain': '.zhihu.com'
        },
    {
        'name':'__snaker__id',
        'value':'8HPJw7gf8gpZUi8b',
        'domain': '.zhihu.com'
        } ]
    time.sleep(3) #把身份卡给网页看的时间
    for cookie in cookies:
        p1.add_cookie(cookie)
    #为模拟浏览器添加已登录后的网址的cookies，跳过登录界面

    p1.get('https://www.zhihu.com/') #跳转到登录界面
    p1.implicitly_wait(4) 
    p1.maximize_window()
    WebDriverWait(p1,1000).until(ec.url_to_be('https://www.zhihu.com/'))
    print('登陆成功！')

    p1.find_element(By.XPATH,'/html/body/div[1]/div/div[3]/header/div/div[1]/div[1]/nav/a[3]').click() #点击热榜按键并切换网页
    switch_page()

    Urls_element = p1.find_elements(By.XPATH,'//div[@class="HotItem-content"]/a')  #selenium的元素定位只能定位到元素属性，不同于xpath功能，不能直接获得链接属性
    Urls = [url.get_attribute('href') for url in Urls_element[:20]]  #get_attribute获得属性

    hot_discussions = []
    for url in Urls:
        p1.get(url)
        time.sleep(5)
        try:
            unfold_button = WebDriverWait(p1, 10).until(    ##点击展开按键才能匹配到;CLASS_NAME不能有空格，有空格必须用css并且改为.;[contains(属性名如text()@class),'部分字母']
                ec.visibility_of_element_located((By.XPATH,"//button[contains(@class,'Button QuestionRichText-more') and contains(text(),'显示全部')]"))
                )    #显示等待还可以用来直接定义变量，确保出现再确保能点击再点
            WebDriverWait(p1,10).until(ec.element_to_be_clickable(unfold_button))
            p1.execute_script("arguments[0].click();", unfold_button)  #js语法无视所有阻挡直接按
            time.sleep(2)
        except:
             print("话题无展开按钮，跳过点击")
        time.sleep(4)  #强制暂停等待加载
        header = WebDriverWait(p1,10).until(ec.visibility_of_element_located((By.CSS_SELECTOR,'#root > div > main > div > div > div:nth-child(10) > div:nth-child(2) > div > div.QuestionHeader-content > div.QuestionHeader-main > h1'))).text  #获取text也不能在语法里用，

        try:
            content = WebDriverWait(p1,10).until(ec.visibility_of_element_located((By.XPATH,'//span[@class="RichText ztext css-1sds6ep"]'))).text.replace('\n', ' ').replace('  ', ' ').strip()
        except:
            print('无具体内容哦~')
            content = '无'
        replys_element = p1.find_elements(By.XPATH,'//span[@class="RichText ztext CopyrightRichText-richText css-1sds6ep"]')
        replys = [reply.text.replace('\n', ' ').replace('  ', '').strip() for reply in replys_element[:10]]  #对每个元素处理之后再放入新列表，去除换行符和空格符

        time.sleep(2)
        try:
            comment_button = WebDriverWait(p1,10).until(ec.element_to_be_clickable((By.XPATH,"//div[@class='QuestionHeader-Comment']/button[contains(text(),'评论')]")))
            p1.execute_script("arguments[0].click();", comment_button)  #等按键可以被点击再生成这个变量并用js强制点击
        except:
            print('无法点击到评论区')
        scroll_page(2)
       
        comments_element = p1.find_elements(By.XPATH,'//div[@class="CommentContent css-1jpzztt"]')  
        comments = [comment.text.replace('\n', ' ').replace('  ', '').strip() for comment in comments_element[:10]]


        hot_discussion = {
            "标题": header,
            "具体内容": content,
            "回答": "///".join(replys),  #把列表拆成用斜杠分割的一段段话，去除括号
            "评论": "///".join(comments)

        }
        hot_discussions.append(hot_discussion)

        time.sleep(random.uniform(8,15))  #知乎敏感肌，最好随地大小睡
        #pprint(hot_discussion,indent=4)
    with open(r'C:\Users\Lst12\Desktop\知乎火热讨论话题.csv', 'w', encoding='utf-8-sig', newline='') as f:
            headers = ['标题', '具体内容', '回答','评论']
            writer = csv.DictWriter(f,headers)
            writer.writeheader()
            writer.writerows(hot_discussions)
