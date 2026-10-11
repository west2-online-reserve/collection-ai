import requests
import re
import os
import json

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
STORAGE_FILE_PATH = os.path.join(FOLDER_PATH, "福大教务处消息.txt")


headers = {
  "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
  "Accept-Encoding": "gzip, deflate, br, zstd",
  "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
  "Cache-Control": "max-age=0",
  "Connection": "keep-alive",
  "Host": "jwch.fzu.edu.cn",
  "Sec-Fetch-Dest": "document",
  "Sec-Fetch-Mode": "navigate",
  "Sec-Fetch-Site": "none",
  "Sec-Fetch-User": "?1",
  "Upgrade-Insecure-Requests": "1",
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36 Edg/152.0.0.0",
  "sec-ch-ua": "\"Chromium\";v=\"152\", \"Not?A_Brand\";v=\"24\", \"Microsoft Edge\";v=\"152\"",
  "sec-ch-ua-mobile": "?0",
  "sec-ch-ua-platform": "\"Windows\""
}

# 两个参数分别是 要爬取的页面数 福大教务处通知第二页的网址上的数字
def get_urls(all_page_num, page_two_num):
    fzu_urls = []
    fzu_urls.append('https://jwch.fzu.edu.cn/jxtz.htm')
    num = 1
    while num < all_page_num:
        fzu_urls.append('https://jwch.fzu.edu.cn/jxtz/' + str(page_two_num + 1 - num) + '.htm')
        num += 1
    return fzu_urls

def get_one_page(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        response.encoding = 'UTF-8-SIG'
        return response.text
    else:
        print(f"请求失败！状态码：{response.status_code}")
        return ''

def trace_accessory(url):
    if len(url) >= 45:
        print("附件请求失败，需要验证码！")
        return "Error"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        response.encoding = 'UTF-8-SIG'
        text = response.text
        if "<li>附件" in text:
            pattern = re.compile(r'href="(.*?)".*?' +
                                 r'_blank">(.*?)</a>】.*?' +
                                 r'getClickTimes\((.*?),(.*?),"wbnewsfile"', re.S)
            raw_inform_list = re.findall(pattern, text)

            accessory_list = []
            accessory_num = 1
            for tuple in raw_inform_list:
                accessory_url = 'https://jwch.fzu.edu.cn' + tuple[0]
                title = tuple[1]
                params = {
                    'wbnewsid': tuple[2],
                    'owner': tuple[3],
                    'type': 'wbnewsfile',
                    'randomid': 'nattach'
                }
                click_times_url = 'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp'
                accessory_response = requests.get(click_times_url, headers=headers, params=params)
                click_times = re.findall(r'"wbshowtimes":(.*?),',accessory_response.text)
                accessory_list.append({'序号': accessory_num, '标题': title, '附件下载地址':accessory_url, '下载次数':click_times})
                accessory_num += 1
            return accessory_list
    else:
        print(f"附件请求失败！状态码：{response.status_code}")
        return 'Error'

def parse_one_page(html):
    pattern = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2}).*?' +
                         r'</span>【(.*?)】.*?' + 
                         r'<a href=".*?(info.*?)".*?' +
                         r'_blank" title="(.*?)">', re.S)
    raw_inform_list = re.findall(pattern, html)
    print(raw_inform_list)
    inform_list = []

    # 对原始的提取数据加工
    for tuple in raw_inform_list:
        title = tuple[3]
        informant = tuple[1]
        date = tuple[0]
        print("详细页地址：", tuple[2])
        detail_url = 'https://jwch.fzu.edu.cn/' + tuple[2]
        accessory_list = trace_accessory(detail_url)
        inform_list.append({'标题': title, '通知人': informant, '日期': date, '正文地址': detail_url, '附件': accessory_list})
    
    return inform_list


fzu_urls = get_urls(13, 214)

inform_list = []

for url in fzu_urls:
    print("正在处理:", url, "...")
    html = get_one_page(url)
    print("获取原始 html 数据成功！")
    print(html[:400])
    page_inform_list = parse_one_page(html)
    inform_list.extend(page_inform_list)

with open(STORAGE_FILE_PATH, 'w', encoding='utf-8') as f:
    json.dump(inform_list, f, ensure_ascii=False, indent = 2)
    print("数据存储完成")