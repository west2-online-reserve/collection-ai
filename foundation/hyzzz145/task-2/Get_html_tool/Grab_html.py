import requests
import os
from lxml import etree
os.chdir(os.path.dirname(os.path.abspath(__file__)))


head = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36 Edg/155.0.0.0"
}
url = 'https://www.zhihu.com/topic/19555513/hot'
resp = requests.get(url,headers=head)
resp.encoding = 'utf-8'
text = resp.text

tree = etree.HTML(resp.text)

html_bytes = etree.tostring(
    tree,
    pretty_print=True,
    encoding=resp.encoding or "utf-8",
    method="html"
)


with open('test.html','wb') as f:
    f.write(html_bytes)
print("over!")