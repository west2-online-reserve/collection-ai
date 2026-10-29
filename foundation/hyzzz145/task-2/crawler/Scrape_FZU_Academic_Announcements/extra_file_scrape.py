import re
import csv
import os
from itertools import islice
import requests

##进入程序文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

f = open('content.csv','r',newline="",encoding='utf-8')
# read = f.read()
# print(read)
csv_text = csv.reader(f)
name = []
url = []
name_re = re.compile(r"附件\d*.(?P<name>.*?)$")
link_re = re.compile(r"附件链接：(?P<link>.*?)$")
for i in islice(csv_text,0,150):
    if "附件名" in str(i):
        name += [name_re.search(i[0]).group("name")]
    if "附件链接" in str(i):
        url += [link_re.search(i[0]).group("link")]

os.chdir('extra_file')
for i in range(len(name)):
    resp = requests.get(url[i])
    with open(name[i],'wb') as m:
        m.write(resp.content)
    print(f"已下载“{name[i]}”")
f.close()