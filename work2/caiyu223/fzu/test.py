import requests
from lxml import html
from urllib.parse import urljoin
import time
import csv


'''
web = requests.get('https://jwch.fzu.edu.cn/info/1086/4402.htm')
tree = html.fromstring(web.content)

a = tree.xpath('//*[@id="vsb_content"]/div/p/span/text()')
b = tree.xpath('/html/body/div[1]/div[2]/div[2]/form/div/div[1]/div/div[1]/h4')
b = '1'
print(a)
def list_to_str(list):
    strs = ''
    for str in list:
        strs += str
    return strs

c = list_to_str(a)

with open('data.csv', 'w', newline='', encoding='utf-8-sig') as file:
    
    
    writer = csv.writer(file, delimiter=';')
    
    writer.writerows(c)
'''

web = requests.get("https://jwch.fzu.edu.cn/system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=16716630" )
ct = web.headers.get('Content-Type','')
print(ct)
with open('download','wb') as file:
    file.write(web.content)