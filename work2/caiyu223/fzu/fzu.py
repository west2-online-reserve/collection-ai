import requests

from lxml import html
from urllib.parse import urljoin
import time
import csv
import pandas as pd
import re

def csv_initialize():
    with open('data.csv','w',newline='', encoding='utf-8-sig') as file:
        pass
        

def file_downlosd(tree1):
    file_urls = tree1.xpath('/html/body/div[1]/div[2]/div[2]/form/div/div[1]/div/ul//li/a/@href')
    file_names = tree1.xpath('/html/body/div[1]/div[2]/div[2]/form/div/div[1]/div/ul//li/a/text()')
    if file_urls:
        i = 0
        for file_url in file_urls:
            with open(f'file/{file_names[i]}','wb') as f:
                file_url = urljoin('https://jwch.fzu.edu.cn',file_url)
                file = requests.get(file_url)
                f.write(file.content)
    time.sleep(1)


#数据清洗
def test_clean(text):
    result = re.sub(r'\s+', '',text )
    result = re.sub(r'[a-zA-Z]', '', result)
    result = re.sub(r'''[^\u4e00-\u9fa50-9，。！？；："'‘'“”（）《》【】]''','',result)
    result = re.sub(r'[0-9]{5,}','',result)
    result = re.findall(r'正文.*加入收藏',result)
    result = list_to_str(result)
    patterns_to_remove = [
        '正文',
        '加入收藏', 
        '字体：大中小分享到：',
        '发布时间：'
        ]
    for pattern in patterns_to_remove:
        result = re.sub(pattern, '', result)

    return result



csv_initialize()
page = 190
final_page = 205
coloums= ['title','time','department','file_dowload_situation','content']
final_content = []

all_hrefs = []

#翻页
while page <= final_page:
    url = f'https://jwch.fzu.edu.cn/jxtz/{page}.htm'
    web = requests.get(url)
    tree = html.fromstring(web.content)
    hrefs = tree.xpath('//ul[@class="list-gl"]/*/a/@href')
    all_hrefs += hrefs
    
    time.sleep(1)
    page += 1

#将列表中字符串合成一整串字符串
def list_to_str(list):
    strs = ''
    for str in list:
        strs += str
    return strs

#补全网址
all_full_hrefs = []
for href in all_hrefs:
    full_href = urljoin(url,href)
    all_full_hrefs.append(full_href)
    
#读取各个网站中文字信息，并存入csv文件
i = 0
print(all_full_hrefs)
for href in all_full_hrefs:
    print(href)
    web_info = requests.get(href)
    web_info.html.render(sleep=1)

    time.sleep(1)
    tree1 = html.fromstring(web_info.content)

    #爬取数据
    text_list = tree1.xpath('//text()')
    title = tree1.xpath('/html/body/div[1]/div[2]/div[2]/form/div/div[1]/div/div[1]/h4/text()')
    post_time = tree1.xpath('//span[@class="xl_sj_icon"]/text()')
    department = tree1.xpath('/html/body/div[1]/div[2]/div[1]/p/a[3]/text()')
    file = tree1.xpath('//ul[@style="list-style-type:none;"]//li//text()')

    print(file)
    
    result = test_clean(list_to_str(text_list))
    #print(result)

    text = {
        'title':list_to_str(title),
        'time':list_to_str(post_time),
        'department':list_to_str(department),
        'file_dowload_situation':list_to_str(file),
        'content':result
            }
    print(text)
    final_content.append(text)
    
    i += 1
    
    
df = pd.DataFrame(final_content)
print(df.to_string)
df.to_csv('data.csv') 
   

