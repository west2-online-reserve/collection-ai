from multiprocessing.connection import answer_challenge
from time import sleep

from selenium import webdriver
from lxml import etree
import time
from selenium.webdriver.common.by import By  # 导包
import requests
from lxml import etree
bro = webdriver.Edge()  # 创建一个新的Edge浏览器实例
topic = ['为什么现在国内各大高校仍选用谭浩强的《C 程序设计》为教材？', '室友想抄我的代码，给还是不给?', '有哪些让你目瞪口呆的 Bug ？', '现实中程序员是怎样飞快敲代码的？', '你的编程能力从什么时候开始突飞猛进？', '有哪些顶级水平的中国程序员？', '被公司卸磨杀驴能把代码删除吗？', 'Bug是如何产生的？', '将 bug 译作「蚆蛒」，将 debug 译作「揥蚆蛒」，音译兼意译，是不是很巧妙？', '王小波的计算机水平有多好？', '怎样评价《数码宝贝》第一部中的泉光子郎的编程水平？', '为什么祖传代码被称为「屎山」？', '学 Python 都用来干嘛的？', '既然有些人喜欢开挂，为啥不开发一款网游，提供编程接口，允许玩家自行用各种软件，同时允许计算机参与计算？', '程序员必须掌握哪些算法？', '为什么美国程序员工作比中国程序员工作轻松、加班少？', '学会了 C 语言真的可以开发出很多东西吗？', '面试被问到不懂的东西，是直接说不懂还是坚持狡辩一下？', '一行代码可以做什么？', '大一一个学期学多少编程算正常?', '为什么不能有把中文想法直接转换成电脑程序的编程呢？', '打字速度对编程的影响大吗？', '如何系统地自学 Python？', '如何用尽可能少的代码画出刘看山？', '写代码没激情怎么办？', '如何入门 Python 爬虫？', '编程究竟难在哪？', '身为程序员碰到最奇葩的需求是怎样的？', '如何评价最近CSDN的行为？', '为什么有人可以在屎一样混乱的几千行代码里加功能？不重构能驾驭屎山吗？', '程序员很闷骚么？', '有哪些算法惊艳到了你？', '雷军写代码水平如何？', '如果让无所不能的神来写代码，是否能写出没有bug的软件？', '大学计算机系最努力的同学都是如何学习的？', '如何用我奶奶也能听懂的话来讲什么是 debug？', '编程新手如何提高编程能力？', '突然意识自己曾经引以为豪的编程其实是一种工具，这是一种悲哀吗？', '我编程写代码被我妈一直说成玩电脑打游戏，我该咋办？']
html_1 = []
import sys

import pandas as pd


def print_progress_bar(iteration, total, prefix='', length=50):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    progress_bar = f'\r{prefix} |{bar}| {percent:.2f}% Complete'

    # Move cursor to the first line
    sys.stdout.write('\033[1A')  # 上移一行
    sys.stdout.write('\033[K')   # 清除行内容
    sys.stdout.write(progress_bar + '\n')  # 写入新的进度条并换行
    sys.stdout.flush()



def create_csv(file_name):
    """
    创建一个新的 CSV 文件，包含指定的列
    :param file_name: CSV 文件名
    :return: None
    """
    columns = ['问题名', '问题内容'] + [f'回答{i}' for i in range(1, 21)]
    df = pd.DataFrame(columns=columns)
    df.to_csv(file_name, index=False, encoding='utf-8-sig')


def add_data_to_csv(file_name, question_name, question_content, answers):
    """
    将数据导入到 CSV 文件中
    :param file_name: CSV 文件名
    :param question_name: 问题名
    :param question_content: 问题内容
    :param answers: 回答列表
    :return: None
    """
    data = {
        '问题名': question_name,
        '问题内容': question_content
    }
    for i in range(1, 21):
        data[f'回答{i}'] = answers[i - 1] if i - 1 < len(answers) else ''

    df = pd.DataFrame([data])
    df.to_csv(file_name, mode='a', header=False, index=False, encoding='utf-8-sig')





def login_qq():
    """
    使用快捷登录的方法登录知乎网站，返回登录后的页面源码
    :return: None
    """
    bro.get('https://www.zhihu.com/question/328444462/answer/1194782618')
    # 访问页面
    time.sleep(2)
    try:
        bro.find_element(By.XPATH,'/html/body/div[1]/div/div[2]/header/div[1]/div[2]/div[2]/div/button').click()
    except:
        print('已弹窗')

    time.sleep(2)
    bro.find_element(By.XPATH,'/html/body/div[5]/div/div/div/div[2]/div/div/div/div[2]/div[1]/div[2]/div/div[3]/span/button[2]').click()  # 点击登录按钮
    time.sleep(2)

    bro.switch_to.window(bro.window_handles[1])  # 切换到新打开的窗口
    bro.switch_to.frame(bro.find_elements(By.TAG_NAME, "iframe")[0])  # 切换到iframe
    time.sleep(0.5)

    bro.find_element(By.CSS_SELECTOR,'.face').click()  # 点击qq登录按钮
    time.sleep(3)

    bro.switch_to.window(bro.window_handles[0])
    bro.get('https://www.zhihu.com/topic/19554298/top-answers')  # 访问页面




def scroll(len_scroll: int,b:int):
    """
    向下滚动页面，获取更多的问题
    :return: None
    """
    for i in range(0, len_scroll, b):  # 每次滚动300像素
        bro.execute_script(f"window.scrollBy(0, {i});")
        time.sleep(0.3)  # 每次滚动后等待页面加载
        bro.execute_script(f"window.scrollBy(0, -{10});")
        time.sleep(0.5)  # 每次滚动后等待页面加载




def get_html(url):
    """
    获取页面源码
    :param url:
    :return:
    """
    # 获取页面源码
    main_response = bro.page_source

    # 解析页面源码
    main_etree = etree.HTML(main_response)

    list1 = main_etree.xpath("/html/body/div[1]/div/main/div/div[1]/div[2]/div[4]/div/div/div/div")
    counts_1 = 1

    for k1 in list1:
        try:
            if not k1.xpath('./div/div/h2/div/a/text()')[0] in topic:
                topic.append(k1.xpath('./div/div/h2/div/a/text()')[0])
                topic_word = k1.xpath('./div/div/h2/div/a/text()')[0]
            else:
                continue
        except:
            print('没有这个问题')
            continue

        print_progress_bar(counts_1, 50, prefix='进度:')
        # 输出日志信息
        print(f"当前步骤: {counts_1} 正在进行的问题名: {topic_word}")


        counts_1 += 1

        # 限制问题数量
        if counts_1 == 50:
            break


        try:
            # 切换网站到列表中的每一个问题
            bro.get("https://" + k1.xpath('./div/div/h2/div/a/@href')[0][2::])
            time.sleep(1.5)
            bro.find_element(By.XPATH,'/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/a').click()
        except:
            print('没有按到按钮')
            continue

        try:
            bro.find_element(By.XPATH,'/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/button').click()
        except:
            print('没有按到文本展开按钮')




        time.sleep(0.5)
        # 滚动页面
        scroll(7000, 200)
        time.sleep(0.5)

        # 获取页面源码
        main_response = bro.page_source
        main_etree = etree.HTML(main_response)

        content_1 = ''


        content_list = main_etree.xpath('/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/div/span/p')
        if len(content_list) != 0:
            for content_word in content_list:
                if len(content_word.xpath('./text()')) > 0:
                    content_1 += content_word.xpath('./text()')[0]
        else:
            print('问题内容不需要展开or没有问题内容')
            content_list = main_etree.xpath('/html/body/div[1]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/div/span')
            if len(content_list) != 0:
                for content_word in content_list:
                    if len(content_word.xpath('./text()')) > 0:
                        content_1 += content_word.xpath('./text()')[0]
            else:
                print('没有问题内容')

        counts = 0
        answer_list1 = []


        list2 = main_etree.xpath('/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div/div/div/div/div[2]/div/div[@class="List-item"]')
        for k2 in list2:

            answer_list = k2.xpath('./div/div/div[2]/span[1]/div/div/span/p')
            answer = ''
            for answer_word in answer_list:
                if len(answer_word.xpath('./text()')) > 0:
                    answer = answer +  answer_word.xpath('./text()')[0]


            answer_list1.append(answer)

            if counts == 21:
                break

            counts += 1

        add_data_to_csv('questions_1.csv', topic_word, content_1, answer_list1)
        print()
        print()
        print()
        sys.stdout.flush()


def main():

    # 创建一个新的 CSV 文件 （如果文件已经存在，会被覆盖）
    create_csv('questions_1.csv')

    # 用qq登录
    login_qq()

    # 初始化进度条
    print_progress_bar(0, 100, prefix='进度:')


    # 等待页面加载
    time.sleep(2)

    # 向下滚动页面
    scroll(10000,200)

    # 获取页面源码
    get_html('https://www.zhihu.com/topic/19554298/top-answers')

    # 关闭浏览器
    bro.quit()






main()



# ['为什么现在国内各大高校仍选用谭浩强的《C 程序设计》为教材？', '室友想抄我的代码，给还是不给?', '有哪些让你目瞪口呆的 Bug ？', '现实中程序员是怎样飞快敲代码的？', '你的编程能力从什么时候开始突飞猛进？', '有哪些顶级水平的中国程序员？', '被公司卸磨杀驴能把代码删除吗？', 'Bug是如何产生的？', '将 bug 译作「蚆蛒」，将 debug 译作「揥蚆蛒」，音译兼意译，是不是很巧妙？', '王小波的计算机水平有多好？', '怎样评价《数码宝贝》第一部中的泉光子郎的编程水平？', '为什么祖传代码被称为「屎山」？', '学 Python 都用来干嘛的？', '既然有些人喜欢开挂，为啥不开发一款网游，提供编程接口，允许玩家自行用各种软件，同时允许计算机参与计算？', '程序员必须掌握哪些算法？', '为什么美国程序员工作比中国程序员工作轻松、加班少？', '学会了 C 语言真的可以开发出很多东西吗？', '面试被问到不懂的东西，是直接说不懂还是坚持狡辩一下？', '一行代码可以做什么？', '大一一个学期学多少编程算正常?', '为什么不能有把中文想法直接转换成电脑程序的编程呢？', '打字速度对编程的影响大吗？', '如何系统地自学 Python？', '如何用尽可能少的代码画出刘看山？', '写代码没激情怎么办？', '如何入门 Python 爬虫？', '编程究竟难在哪？', '身为程序员碰到最奇葩的需求是怎样的？', '如何评价最近CSDN的行为？', '为什么有人可以在屎一样混乱的几千行代码里加功能？不重构能驾驭屎山吗？', '程序员很闷骚么？', '有哪些算法惊艳到了你？', '雷军写代码水平如何？', '如果让无所不能的神来写代码，是否能写出没有bug的软件？', '大学计算机系最努力的同学都是如何学习的？', '如何用我奶奶也能听懂的话来讲什么是 debug？', '编程新手如何提高编程能力？', '突然意识自己曾经引以为豪的编程其实是一种工具，这是一种悲哀吗？', '我编程写代码被我妈一直说成玩电脑打游戏，我该咋办？']
# ['为什么现在国内各大高校仍选用谭浩强的《C 程序设计》为教材？', '室友想抄我的代码，给还是不给?', '有哪些让你目瞪口呆的 Bug ？', '现实中程序员是怎样飞快敲代码的？', '你的编程能力从什么时候开始突飞猛进？', '有哪些顶级水平的中国程序员？', '被公司卸磨杀驴能把代码删除吗？', 'Bug是如何产生的？', '将 bug 译作「蚆蛒」，将 debug 译作「揥蚆蛒」，音译兼意译，是不是很巧妙？', '王小波的计算机水平有多好？', '怎样评价《数码宝贝》第一部中的泉光子郎的编程水平？', '为什么祖传代码被称为「屎山」？', '学 Python 都用来干嘛的？', '既然有些人喜欢开挂，为啥不开发一款网游，提供编程接口，允许玩家自行用各种软件，同时允许计算机参与计算？', '程序员必须掌握哪些算法？', '为什么美国程序员工作比中国程序员工作轻松、加班少？', '学会了 C 语言真的可以开发出很多东西吗？', '面试被问到不懂的东西，是直接说不懂还是坚持狡辩一下？', '一行代码可以做什么？', '大一一个学期学多少编程算正常?', '为什么不能有把中文想法直接转换成电脑程序的编程呢？', '打字速度对编程的影响大吗？', '如何系统地自学 Python？', '如何用尽可能少的代码画出刘看山？', '写代码没激情怎么办？', '如何入门 Python 爬虫？', '编程究竟难在哪？', '身为程序员碰到最奇葩的需求是怎样的？']