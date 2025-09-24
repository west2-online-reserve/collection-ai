# 官方库
import requests
from lxml import etree
import time
import random
# 私有库
import csv_making
import attachment



# 福州大学教务处 页面获取处理
def get_html(url):
    """
    # 福州大学教务处 页面获取处理
    :param url: 获取的网页（可以是每一个单独的主页）
    :return: 返回的数据为福大的主页html （etree类型）
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
    }
    response = requests.get(url, headers=headers)
    html_etree = etree.HTML(response.text)
    return html_etree

# 获取福大教务处每一页教务处的网址
def get_all_page_html(html_etree):
    """

    :param html_etree: 需要传入福大教务处主页的html（etree类型）
    :return: 返回每张页面的网址（类型为list 以列表的形式进行返回 为正序 页面）
    """
    numbers = ""
    html_list = ["https://jwch.fzu.edu.cn/jxtz.htm"]
    html_all_str =  html_etree.xpath("/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[2]/div[1]/div/span[1]/span[4]/a/@href")[0]
    for char in html_all_str:
        if char.isdigit():
            numbers += char
    numbers1 = int(numbers) + 1
    for i in range(numbers1 - 1 , 0 , -1) :
        html_list.append("https://jwch.fzu.edu.cn/jxtz/" + str(i) + ".htm")
    return html_list

# 爬取每一张页面对应的 数据
def get_info(html_list):
    """

    :param html_list: 网页的列表
    :return: None
    """
    for html in html_list:
        html_etree = get_html(html)

        list1 = html_etree.xpath("/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul/li")

        for fzu in list1:
            time.sleep(random.uniform(0.1, 0.25))
            # 教务系统的通知人
            fzu_human = fzu.xpath("./text()")[1]
            fzu_human = fzu_human.encode('iso-8859-1').decode('utf-8')
            fzu_human = fzu_human.replace("】","")
            fzu_human = fzu_human.replace("【","")
            fzu_human = fzu_human.strip()

            # 教务系统的通知时间
            fzu_time = fzu.xpath("./span/font/text()")
            if len(fzu_time) == 0:
                fzu_time = fzu.xpath("./span/text()")
            fzu_time = fzu_time[0].strip()
            # fzu_time = fzu_time.encode('iso-8859-1').decode('utf-8')
            # 教务系统的标题
            fzu_body = fzu.xpath("./a/text()")[0]
            fzu_body = fzu_body.encode('iso-8859-1').decode('utf-8')
            # 通知详情地址
            fzu_header = "https://jwch.fzu.edu.cn/" + fzu.xpath("./a/@href")[0]
            fzu_header = fzu_header.encode('iso-8859-1').decode('utf-8')

            # 调用附件模块 爬取附件
            fzu_header_etree = get_html(fzu_header)
            num , file_name ,file_download = attachment.get_istrue(fzu_header_etree)

            # print(num)
            sum = 0
            name = ""

            if num != 0:
                for i in range(num):
                    sum += file_download[0]
                    name += file_name[0]


            csv_making.write_csv(fzu_human, fzu_body, fzu_time ,fzu_header,sum,name)


def main():
    csv_making.new_csv()
    get_info(get_all_page_html(get_html("https://jwch.fzu.edu.cn/jxtz.htm")))

main()














