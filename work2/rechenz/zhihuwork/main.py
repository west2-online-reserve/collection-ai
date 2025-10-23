from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.common import exceptions
import time
import csv
import json
import bs4
import re


def main():
    service = EdgeService()
    options = Options()  # 先声明一个options变量
    # 隐藏自动化控制标头
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option(
        'excludeSwitches', ['enable-automation'])  # 隐藏自动化标头
    options.add_argument('--ignore-ssl-errosr')  # 忽略ssl错误
    options.add_argument('--ignore-certificate-errors')  # 忽略证书错误
    options.add_experimental_option("useAutomationExtension", False)
    # prefs = {
    #     'download.default_directory': '文件夹路径',  # 设置文件默认下载路径
    #     "profile.default_content_setting_values.automatic_downloads": True  # 允许多文件下载
    # }
    # options.add_experimental_option("prefs", prefs)  # 将prefs字典传入options

    browser = webdriver.ChromiumEdge(options)
    browser.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
                })
            """
        },
    )
    browser.get('https://www.zhihu.com/topic/19554298/top-answers')
    # input()
    # cookies = browser.get_cookies()
    # with open("AIsolution/work2/zhihuwork/cookies.json", "w") as file:
    #     json.dump(cookies, file)
    with open("AIsolution/work2/zhihuwork/cookies.json", "r") as file:
        cookies = json.load(file)
        for cookie in cookies:
            browser.add_cookie(cookie)
    browser.refresh()
    # exit()

    def get_element(browser, xpath):
        try:
            browser.find_element(By.XPATH, xpath)
            return True
        except exceptions.NoSuchElementException:
            return False

    with open("AIsolution/work2/zhihuwork/zhihu.csv", "w", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["问题名", "问题具体内容", "回答信息"])
        num = 2
        count = 1
        while count <= 20:  # 从问题开始
            time.sleep(3)
            print(f"第{count}个问题")
            # if num == 19:
            #     print(1)
            check = get_element(
                browser, f'//*[@id="TopicMain"]/div[4]/div/div/div/div[{num}]/div/div/h2/div/a')
            flag = True
            if check:
                curelement = browser.find_element(
                    By.XPATH, f'//*[@id="TopicMain"]/div[4]/div/div/div/div[{num}]/div/div/h2/div/a')
            else:
                flag = False
                curelement = browser.find_element(
                    By.XPATH, f'//*[@id="TopicMain"]/div[4]/div/div/div/div[{num}]/div/div/h2/span/a')
                num += 1
            browser.execute_script(
                "arguments[0].scrollIntoView();", curelement)
            if flag == False:
                continue
            surl = curelement.get_attribute("href")
            if surl == None:
                print("error")
                break
            browser.get(surl)
            problemname = ''
            problemcontent = ''
            answercontent = ''
            li = []
            scurelement = browser.find_element(
                By.XPATH, '//*[@id="root"]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/h1')
            problemname = scurelement.text
            check = get_element(
                browser, '/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div')
            if check != False:
                checkbutton = re.search('显示全部', browser.page_source)
                if checkbutton:
                    scurelement = browser.find_element(
                        By.XPATH, '/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/button')
                    scurelement.click()
                    scurelement = browser.find_element(
                        By.XPATH, '/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/div/span/span[1]')
                    problemcontent = scurelement.text
                else:
                    scurelement = browser.find_element(
                        By.XPATH, '/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/div/span/span[1]')
                    problemcontent = scurelement.text
            scurelement = browser.find_element(
                By.CSS_SELECTOR, '#content > span.RichText.ztext.css-oqi8p3')
            problemcontent = scurelement.text
            li.append(problemname)
            li.append(problemcontent)
            time.sleep(1)
            show = browser.find_element(
                By.XPATH, '/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[4]/a')
            #              /html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[4]/a
            show.click()
            firstanwser = browser.find_element(
                By.XPATH, '/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[2]/div/div/div[2]/span[1]/div/div/span')
            answercontent = firstanwser.text
            li.append(answercontent)
            s = 2
            i = 3
            last = False
            while s <= 10:  # 10个回答
                print(f"第{s}个回答")
                checkcur = get_element(
                    browser, f'/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[{i}]/div/div/div[2]/span[1]/div/div/span')
                if checkcur == False:
                    i += 1
                    if last == True:
                        s += 1
                    last = True
                    continue
                scurelement = browser.find_element(
                    By.XPATH, f'/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[{i}]/div/div/div[2]/span[1]/div/div/span')
                '''
                /html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[3]/div/div/div[2]/span[1]/div/div/span
                /html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[4]/div/div/div[2]/span[1]/div/div/span
                /html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[3]/div/div/div[2]/span[1]/div/div/span
                /html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[4]/div/div/div[2]/span[1]/div/div/span
                /html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[6]/div/div/div[2]/span[1]/div/div/span
                '''
                answercontent = scurelement.text
                li.append(answercontent)
                browser.execute_script(
                    "arguments[0].scrollIntoView();", scurelement)
                i += 1
                s += 1
                time.sleep(2)
            writer.writerow(li)
            num += 1
            count += 1
            browser.back()
            browser.back()


if __name__ == '__main__':
    main()
