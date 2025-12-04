import requests
import re
from lxml import etree
import pandas as pd
from pathlib import Path

Notices = []

def get_click_times(response):
    html = etree.HTML(response.text)
    t = html.xpath("//ul[@style]/li/span/script/text()")
    click_times = []
    for i in t:
        p = re.findall(r"(?<=[,\(\"])\w+(?=[,\)\"])", i)
        params = dict(zip(["wbnewsid","owner","type","randomid"],p))
        get_response = requests.get("https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp",params=params)
        click_times.append(re.findall(r"(?<=\"wbshowtimes\":)\d+(?=,)",get_response.text)[0])
    return click_times

def select_useful(pattern, list):
    ans = []
    for n in list:
        a = re.findall(pattern, n)
        if len(a) != 0:
            ans.append(a[0])
    return ans


def get_notice(response):

    notices = []

    html = etree.HTML(response.text)

    # 获取通知人
    t = html.xpath('//ul[@class = "list-gl"]/li/text()')
    notifiers = select_useful(r"\w+", t)

    # 获取标题
    titles = html.xpath('//ul[@class = "list-gl"]/li/a[@title]/text()')

    # 获取时间
    t = html.xpath(
        '//ul[@class = "list-gl"]/li/span[@class = "doclist_time"]/text()|'
        '//ul[@class = "list-gl"]/li/span[@class = "doclist_time"]/font/text()'
    )
    dates = select_useful(r"\d{4}-\d{2}-\d{2}", t)

    # 获取详情链接
    urls = html.xpath('//ul[@class = "list-gl"]/li/a[@href]/@href')
    for i in range(0, len(urls)):
        urls[i] = "https://jwch.fzu.edu.cn/" + urls[i]

    for notifier, title, date, url in zip(notifiers, titles, dates, urls):
        notices.append({"notifier": notifier, "title": title, "date": date, "url": url})

    return notices

# 获取附件

def get_attatchments(notices):
    for i in range(0,len(notices)):
        response = requests.get(notices[i]["url"])
        response.encoding = "utf-8"
        html = etree.HTML(response.text)
        attachment_cnt = len(html.xpath("//ul[@style]/li[contains(text(),'附件')]/span/script/text()"))
        print(f"正在获取 {notices[i]["title"]} 的附件,{i}/{len(notices)}")
        if (attachment_cnt != 0):
            title = html.xpath("//ul[@style]/li[contains(text(),'附件')]/a/text()")
            download_times = get_click_times(response)
            urls = html.xpath("//ul[@style]/li[contains(text(),'附件')]/a/@href")
            for j in range(0,len(urls)):
                urls[j] = "https://jwch.fzu.edu.cn" + urls[j]
        else:
            title = ["无"]
            download_times = [0]
            urls = ["无"]
        attachment = {"attachment_cnt":attachment_cnt,"file_title":title,"download_times" : download_times,"file_url" : urls}
        notices[i].update(attachment)

            

response = requests.get("https://jwch.fzu.edu.cn/jxtz.htm")

response.encoding = "utf-8"

Notices += get_notice(response)

for i in range(181,206)[::-1]:
    response = requests.get(f"https://jwch.fzu.edu.cn/jxtz/{i}.htm")
    response.encoding = "utf-8"
    Notices += get_notice(response)

get_attatchments(Notices)

csv_file_path = Path(__file__).parent / "fzu_jwch_notice.csv"

df = pd.DataFrame(Notices)
df.to_csv(csv_file_path,index=False,encoding="utf-8-sig",sep=",")
