
import json
import time
from selenium import webdriver
from time import sleep
from selenium.webdriver import ChromeOptions
from selenium.common.exceptions import NoSuchElementException
import csv

def Get_Cookise():
    driver = webdriver.Edge()
# 记得写完整的url 包括http和https
    driver.get('https://www.zhihu.com')
    # 程序打开网页后20秒内 “手动登陆账户”
    time.sleep(20)
    with open('cookies.txt', 'w') as f:
        # 将cookies保存为json格式
        f.write(json.dumps(driver.get_cookies()))

    driver.close()
def Login():
    # 设置 UA
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    option = ChromeOptions()
    option.add_experimental_option('excludeSwitches', ['enable-automation'])
    driver = webdriver.Chrome(options=option)
    driver.get('https://www.zhihu.com')
    sleep(2)
    driver.delete_all_cookies()
    with open('cookies.txt', 'r') as f:
        # 使用json读取cookies
        cookies_list = json.load(f)
        # 方法1 将expiry类型变为int
        for cookie in cookies_list:
            # 并不是所有cookie都含有expiry 所以要用get方法来获取
            if isinstance(cookie.get('expiry'), float):
                cookie['expiry'] = int(cookie['expiry'])
            driver.add_cookie(cookie)
    driver.refresh()
    CrawlQuestions(driver)

    #爬取问题
def CrawlQuestions(driver):
    driver.get('https://www.zhihu.com/topic/19554298/top-answers')
    sleep(5)
    for o in range(3):
          driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
          sleep(1)

    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    l=[]
    a=0
    ti = []
    for i in range(2,100):
        smooth_scroll_step(driver,100)
        try:
            element = driver.find_element_by_css_selector(f"#TopicMain > div.ListShortcut > div > div > div > div:nth-child({i}) > div > div > h2  a")
            driver.execute_script("arguments[0].scrollIntoView();", element)

        except NoSuchElementException as e:
            try:
                print("捕获到异常:")
                # TopicMain > div.ListShortcut > div > div > div > div:nth-child(39) > div > div > h2
                element = driver.find_element_by_css_selector(
                    f"# TopicMain > div.ListShortcut > div > div > div > div:nth-child({i}) > div > div > h2 > span > a")
            except NoSuchElementException:
                sleep(15)
                i-1
                print("跳过")
                continue
        else:
            print("定位到元素")

        title= element.text
        url=element.get_attribute('href')
        print(title,url)
        if title in ti:
                print(f"'{title}' 在列表中。")
        else:
                print(f"'{title}' 不在列表中。")
                ti.append(title)
                l.append(url)
                smooth_scroll_step(driver,-500)
                smooth_scroll_step(driver,1000)
                smooth_scroll_step(driver, -500)

                element.click()
                sleep(2)
                CrawlAnswers(url, driver, title)
        sleep(0.1)
        print(len(ti))
        if len(ti)==50:
            break
        sleep(0.5)
        smooth_scroll_step(driver, 100)



    # for n in range(9,len(l)):
    #     url=l[n]
    #     title=ti[n]
    #     CrawlAnswers(url,driver,title)
    #     sleep(0.5)
def CrawlAnswers(url,driver,title):
    driver.get(url)
    sleep(5)
    # 定位所有class="RichContent-inner"的元素
    # rich_content_elements = driver.find_element("css selector", ".RichText.ztext.css-ob6uua")
    # print(rich_content_elements.text)


    try:
        element = driver.find_element_by_css_selector("#root > div > main > div > div > div:nth-child(10) > div:nth-child(2) > div > div.QuestionHeader-content > div.QuestionHeader-main > div:nth-child(7) > div > div > div > div > span")
        Question = element.text.replace("\n", "")
        print(Question)
    except NoSuchElementException:

        print("change")
        try:

            button = driver.find_element_by_css_selector(
                "#root > div > main > div > div > div:nth-child(10) > div:nth-child(2) > div > div.QuestionHeader-content > div.QuestionHeader-main > div:nth-child(7) > div > div > div > button")
            button.click()
            sleep(0.1)
            # element = driver.find_element_by_css_selector(
            #     "#root > div > main > div > div > div:nth-child(10) > div:nth-child(2) > div > div.QuestionHeader-content > div.QuestionHeader-main > div:nth-child(7) > div > div > div > div > span")
            rich_content_elements = driver.find_element("css selector", ".RichText.ztext.css-ob6uua")
            print(rich_content_elements.text)
            Question=rich_content_elements.text



        except NoSuchElementException:
            print("无信息")
            Question="空"
    sleep(0.5)
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    sleep(2)
    button = driver.find_element_by_xpath(" /html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[4]/a")
    button.click()
    sleep(1)
    for i in range(3):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        sleep(0.5)

    driver.execute_script("window.scrollTo(0, 0);")
    sleep(1)
    print("开始爬取")
    # for i in range(7):
    #     driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    #     sleep(1)


    for a in range(10):
        smooth_scroll_step(driver, 500)
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        smooth_scroll_step(driver, -500)
    QNA = [title, Question]

    for h in range(3,23):
        try:

            t="QuestionAnswers-answers"
            ps= driver.find_element_by_xpath(f"/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div/div/div/div/div[2]/div/div[{h}]/div/div/div[2]/span[1]/div/div/span")
        except NoSuchElementException:
            print(f"{h}异常")
            continue
        driver.execute_script("arguments[0].scrollIntoView();", ps)
        a=ps.text.replace("\n", "")
        print(f"回答：{a}")

        QNA.append(a)
        print(QNA)
        print(f"{h-2}爬取完毕")

        smooth_scroll_step(driver,500)
        sleep(0.8)

    print(QNA)
    writer.writerow(QNA)
    driver.back()
    sleep(2)
    driver.back()

def smooth_scroll_step(driver, pixels, step=50, delay=0.1):
    # 计算滚动次数
    steps = abs(pixels) // step
    direction = 1 if pixels > 0 else -1

    for _ in range(steps):
        driver.execute_script(f"window.scrollBy(0, {direction * step});")
        time.sleep(delay)


if __name__ == '__main__':
    # Get_Cookise()

    # 引用csv模块。
    csv_file= open('zhihu.csv', 'a', newline='', encoding='utf-8')
    # 调用open()函数打开csv文件，传入参数：文件名“demo.csv”、写入模式“w”、newline=''、encoding='gbk'
    writer = csv.writer(csv_file)
    #用csv.writer()函数创建一个writer对象。
     # writer.writerow(['通知人', '标题','日期','详情链接','附件名','附件下载次数','附件链接','附件名','附件下载次数','附件链接','附件名','附件下载次数','附件链接','附件名','附件下载次数','附件链接','附件名','附件下载次数','附件链接','附件名','附件下载次数','附件链接'])
    # l1=['问题', '问题具体信息']
    # l=['回答']
    # for n in range(0,20):
    #       l1.append(l[0])
    #
    # print(l1)
    # writer.writerow(l1)
    Login()



