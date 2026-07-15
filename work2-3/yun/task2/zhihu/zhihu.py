from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options #用于设置浏览器选项
from selenium.webdriver.common.by import By#导入By类，用于指定元素定位方式
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
import random
import json
import re
import jsonpath
def setup():#创建浏览器对象的函数
    
    cookies =''#在这里粘贴cookie字符串，注意要用单引号括起来，并且将双引号替换为单引号
        
    options = Options()
    options.add_argument("--no-sandbox")  #禁用沙盒模式（增加系统兼容性）
    #options.add_argument("--headless")  #启用无头模式（不显示浏览器界面）
    options.add_experimental_option(name='detach', value=True)  #保持浏览器打开状态
    options.add_experimental_option("excludeSwitches", ["enable-automation"])#去除浏览器自动化控制的提示
    options.add_experimental_option('useAutomationExtension', False)#禁用浏览器自动化扩展
    options.add_argument('user_agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36')#设置浏览器的User-Agent字符串，模拟正常用户访问
    driver = webdriver.Edge(service=Service('msedgedriver.exe'),options=options)  #创建Edge浏览器对象
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    """
})#通过执行CDP命令，修改浏览器的navigator.webdriver属性，使其返回undefined，进一步隐藏自动化控制的痕迹
    driver.implicitly_wait(10)#设置隐形式等待时间，单位为秒，在查找元素时，如果元素没有立即出现，Selenium会等待一段时间，直到元素出现或者超时
    driver.get("https://www.zhihu.com/")#打开知乎首页
    for i in cookies.split("; "):#将cookie字符串按照"; "分割成一个个cookie，然后遍历每个cookie
        name, value = i.split("=", 1)
        driver.add_cookie({"name": name, "value": value})#将cookie字符串分割成键值对，并使用add_cookie方法将每个cookie添加到浏览器中
    driver.refresh()#刷新页面，使添加的cookie生效
    return driver
def switch_to_new_window(driver, x, oldurl=None):#切换到新打开的窗口或标签页的函数
    handles = list(driver.window_handles)  # 获取所有窗口或标签页的句柄
    driver.switch_to.window(handles[x])  # 切换到指定窗口或标签页
    print(f"从{oldurl}切换到{driver.current_url}")#打印切换前后的URL
header={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "cookie":'',#在这里粘贴cookie字符串，注意要用单引号括起来，并且将双引号替换为单引号
    "Content-Type": "application/json"
}
driver = setup()#调用setup函数创建浏览器对象
wait = WebDriverWait(driver, 10)  # 创建WebDriverWait对象，等待时间为10秒

driver.get("https://www.zhihu.com/topics")#打开知乎话题广场

time.sleep(random.randint(2, 5))#等待页面加载完成
n=0
with open("zhihu.csv","w",encoding="utf-8") as f:#将问题内容保存到zhihu.csv文件中
    while n<=15:#爬取前15个话题的内容    
        topics=driver.find_elements(By.XPATH,value=f"//div[@class='zh-general-list clearfix']//strong")
        print(f"正在爬取第{n+1}个话题：{topics[n].text}......")#打印正在爬取的话题名称
        f.write(f"第{n+1}个话题: {topics[n].text}\n")#先写入话题与个数
        topics[n].click()#点击话题进入话题详情页
        
        switch_to_new_window(driver,-1,driver.current_url)#切换到新打开的窗口或标签页
        time.sleep(random.randint(3,5))#等待页面加载完成
        try:
            wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "等待回答"))).click()
        except:
            wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "等待回答"))).click()
        time.sleep(random.randint(1,3))#等待页面加载完成
        try:
            wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='List']//a[@href]"))).click()#点击第一个问题进入问题详情页
        except:
            print("系统繁忙，请稍后再试！直接关闭当前窗口，继续爬取下一个话题！")
            n+=1
            driver.close()
            switch_to_new_window(driver,0)#切换到第一个标签页,即话题广场
            time.sleep(random.randint(2, 5))#等待页面加载完成
            continue
        switch_to_new_window(driver,-1,driver.current_url)#切换到新打开的窗口或标签页
        title=driver.find_element(By.XPATH,value="//*[@id='root']/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/h1").text
        try:
            driver.find_element(By.XPATH,value="//div[@class='QuestionHeader-main']//button").click()#点击显示全部按钮，展开问题描述内容
            question=driver.find_element(By.XPATH,value="//div[@class='QuestionHeader-main']//span[@id='content']").text#获取问题的描述内容
        except:
            print("未能找到显示全部按钮！")
            question = "没有问题描述！"
        try:
            answer_id=re.search(r'\d+',driver.current_url).group()#从当前URL中提取问题的ID
            commit=requests.get(f"https://www.zhihu.com/api/v4/comment_v5/questions/{answer_id}/root_comment?order_by=score&limit=20",headers=header)#通过知乎的API接口获取问题的回答内容，并将返回的JSON数据解析为Python字典对象
            commits=jsonpath.jsonpath(json.loads(commit.text), "$..content")#使用jsonpath库从解析后的数据中提取所有回答的内容，并将其保存在commits列表中
            clean_commits=[re.sub(r'<.*?>', '', i) for i in commits]#使用正则表达式去除回答内容中的HTML标签，得到干净的文本内容，并将其保存在clean_commits列表中
        except:
            print("未能获取到评论！")
            clean_commits = ["没有评论"]
        time.sleep(random.randint(1,3))#等待页面加载完成

        for i in range(5):#通过循环多次滚动页面，确保页面上加载了足够的内容
            driver.execute_script("window.scrollTo(0, -100);")#将页面向上滚动100像素，以触发页面的加载机制，确保更多内容被加载出来
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")#将页面滚动到最底部，以加载更多内容
            content=driver.find_elements(By.ID,value="content")
            if len(content)>10:#如果页面上已经得到10条内容，就跳出循环
                break
        
        f.write(f"问题标题: {title}\n\n")#先写入问题标题
        f.write(f"问题描述: {question}\n\n")#写入问题描述
        f.write("一些评论\n")
        for i in clean_commits:#写入每条回答的内容
            f.write(f"{clean_commits.index(i)+1}. {i}\n")#写入每条回答的内容，并在每条回答后面写入回答的序号
        num=1
        f.write(f"-------------------回答内容-------------------\n")#写入页面上加载的回答内容
        for i in content[1:]:
            f.write(f"-------------------第{num}条回答-------------------\n")#写入每条回答的内容，并在每条回答后面写入回答的序号
            f.write(i.text+"\n")
            num+=1
        driver.close()#关闭当前窗口或标签页
        switch_to_new_window(driver,-1)
        driver.close()#关闭当前窗口或标签页
        switch_to_new_window(driver,0)#切换到第一个标签页,即话题广场
        time.sleep(random.randint(2, 5))#等待页面加载完成
        n+=1
driver.quit()#关闭浏览器对象，结束程序运行