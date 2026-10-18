import requests,os,time,json,selenium,csv,traceback
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from lxml import etree
from Grab_html import html_grab
#进入项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#初始化
problem = []
content = []
answer = []
discuss_run = True

#函数
def findelement_xpath(xpath:str):
        global element
        wait.until(
             EC.presence_of_element_located((By.XPATH,xpath))
        )
        element = drive.find_element(By.XPATH,xpath)
        #居中
        time.sleep(0.1)
        if 'bottom' in xpath.split("/")[-1]:
            drive.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});",
            element
            )
        else:
            drive.execute_script(
        "arguments[0].scrollIntoView({block: 'center'});",
        element
        )




#网站
#知乎
#url = "https://www.zhihu.com/"
#话题广场
url = "https://www.zhihu.com/topics"

#drive
drive = webdriver.Edge()
drive.get(url)
#webwait
wait = WebDriverWait(drive,5)
#--------------------------
# # 2. 读取你保存的 cookies
# with open("cookie.json", "r", encoding="utf-8") as f:
#     cookies = json.load(f)
# print(cookies)
# # 3. 逐个添加
# for cookie in cookies:
#     try:
#         drive.add_cookie(cookie)
#     except Exception as e:
#         print(f"跳过 cookie: {cookie.get('name')}, 原因: {e}")
#--------------------
#读取json
with open("cookie.json",'r',encoding='utf-8') as f:
    cookies = json.load(f)
#逐个添加
for cookie in cookies:
    try:
        drive.add_cookie(cookie)
    except Exception as e:
        print(f"出现错误{e}")
drive.refresh()
element = drive.find_element(By.XPATH,'/html/body/div[3]/div[1]/div/div/div[2]/div/div[1]/div/a[1]')
hot_topic_url = element.get_attribute('href')
drive.get(hot_topic_url)
time.sleep(3)

#爬取html
# html = drive.page_source
# html_grab(html)

#全屏
drive.fullscreen_window()
#爬取问题
# breakpoint()
i = 2
while len(problem) <= 10:
    print(f"i={i}")
    print(f'probelm = {len(problem)}')
    print(f'content = {len(content)}')
    print(f'discuss = {len(answer)}')
    
    #模块xpath
    XPATH = f'//*[@id="TopicMain"]/div[4]/div[2]/div/div/div[{i}]'
    #问题
    XPATH_QUESTION = XPATH + '/div/div/h2/h2/div/a'
    #正文
    XPATH_CONTENT = XPATH +'/div/div/div/span/div/button[1]'
    # 根据正文是否折叠，选择对应的讨论按钮 XPath。
    XPATH_RICH_CONTENT = XPATH + '/div/div/div'
    element = drive.find_element(By.XPATH, XPATH_RICH_CONTENT)
    if 'RichContent is-collapsed' in element.get_attribute('class'):
        XPATH_DISCUSS_CLICK = XPATH + '/div/div/div/div[2]/div/button[1]'
    else:
        XPATH_DISCUSS_CLICK = XPATH + '/div/div/div[2]/div[2]/div/button[1]'
    try:
        print(f"--------第{i-1}篇---------")
        #点击阅读全文
        findelement_xpath(XPATH_CONTENT)
        element.click()
        print("已点击阅读更多")
        #正文xpath
        XPATH_CONTENT = XPATH + '/div/div/div[2]/span[1]/div/div/span/span[1]'
        #提取问题
        findelement_xpath(XPATH_QUESTION)
        problem.append(element.text.replace("【",'').replace('】',''))
        print("已添加问题列表")
        #提取正文
        findelement_xpath(XPATH_CONTENT)
        content.append(element.text)
        print('已添加正文列表')
        #点击讨论按钮
        # 先滚到当前文章卡片底部，让讨论操作区露出来。
        for attempt in range(3):
            try:
                card = wait.until(EC.presence_of_element_located((By.XPATH, XPATH)))
                drive.execute_script("arguments[0].scrollIntoView({block: 'end'});", card)
                time.sleep(0.3)
                discuss_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, XPATH_DISCUSS_CLICK))
                )
                drive.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});",
                    discuss_button
                )
                # 滚动可能触发页面重绘；重新定位 XPath 对应的按钮再点击。
                discuss_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, XPATH_DISCUSS_CLICK))
                )
                discuss_button.click()
                print("已点击评论")
                break
            except StaleElementReferenceException:
                if attempt == 2:
                    raise
                time.sleep(0.2)
        time.sleep(1)
        #提取评论
        discuss_part = []
        #抓取讨论运行循环
        discuss_run = True
        for o in range(1,11):
            try:
                print(f"第{o}条")
                if discuss_run:
                    XPATH_JUDGE_DISCUSS_2 = XPATH + '/div/div/div[2]'
                    findelement_xpath(XPATH_JUDGE_DISCUSS_2)
                    print(element.get_attribute('style'))
                    if element.get_attribute('style') == 'overflow: unset;':
                        XPATH_DISCUSS = XPATH + f"/div/div/div[2]/div/div/div[2]/div[2]/div/div[{o}]/div/div[2]/div[2]"
                    else:
                        XPATH_DISCUSS = XPATH + f'/div/div/div[3]/div/div/div[2]/div[2]/div/div[{o}]/div/div[2]/div[2]'
                    print(XPATH_DISCUSS)
                    findelement_xpath(XPATH_DISCUSS)
                    print(element.text)
                    discuss_part.append(element.text)
                    print("已添加此评论")
                else:
                    break
            except Exception as e:
                print('少于10条')
                discuss_run = False
            #加入大列表
        answer.append(discuss_part)
        discuss_part = []
        print("已加入评论大列表")

    except Exception as e:
        print("异常类型：", type(e).__name__)
        print("异常内容：", repr(e))
        traceback.print_exc()
    #处理多出数据情况
    l_p = len(problem)
    l_c = len(content)
    l_a = len(answer)
    if (l_p != l_c) or (l_p != l_a) or (l_c != l_a):
        l_max = max(l_p,l_a,l_c)
        for o in [problem,content,answer]:
            if len(o) == l_max:
                o.pop(-1)
        print("已处理数据异常")

    # print(problem[i-2])
    # print(content[i-2])
    # print(answer[i-2])
    i += 1
    time.sleep(0.3)
    # input(f"{i}输入以继续")
#写入
with open('ZhiHu_topice.csv','w',encoding='utf-8',newline="") as f:
    f.write("")
f = open('ZhiHu_topice.csv','a',encoding='utf-8',newline="")
writer = csv.writer(f)
for i in range(len(problem)):
    writer.writerow([f"问题{i+1}：{problem[i]}"])
    writer.writerow([f"正文：{content[i]}"])
    writer.writerow(["回答："])
    for o in range(len(answer[i])):
        writer.writerow([f"\t{o+1}，{answer[i][o]}"])
    writer.writerow(["\n"])
    print(f"第{i+1}个问题已写入")
f.close






    # //*[@id="TopicMain"]/div[4]/div[2]/div/div/div[2]/div/div/h2/h2/div/a
    # //*[@id="TopicMain"]/div[4]/div[2]/div/div/div[3]/div/div/h2/h2/div/a
    #/html/body/div[1]/div/main/div/div[1]/div[2]/div[4]/div[2]/div/div/div[2]/div/div/div/span/div/div/span/span[1]
    #/html/body/div[1]/div/main/div/div[1]/div[2]/div[4]/div[2]/div/div/div[3]/div/div/div/span/div/div/span/span[1]
    #//*[@id="TopicMain"]/div[4]/div[2]/div/div/div[2]/div/div
    #//*[@id="TopicMain"]/div[4]/div[2]/div/div/div[2]/div/div/div/span/div/button
    #//*[@id="TopicMain"]/div[4]/div[2]/div/div/div[2]/div/div/div[2]/div[2]/div/button[1]
    #//*[@id="TopicMain"]/div[4]/div[2]/div/div/div[2]/div/div/div[3]/div/div/div[2]/div[2]/div/div[1]/div/div[2]/div[2]
    #//*[@id="TopicMain"]/div[4]/div[2]/div/div/div[2]/div/div/div[3]/div/div/div[2]/div[2]/div/div[2]/div/div[2]/div[2]

