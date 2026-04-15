import requests
import jsonpath
import json
import time
import os
API_URL = "https://wiki.biligame.com/blhx/api.php"
OUTPUT_DIR = "name_you_like"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0',
    'Accept-Encoding': 'gzip',
}
CATEGORY_PREFIX_MAP = {
    "Category:方案舰娘": "PR",
}
params = {
    "action": "query",
    "list": "categorymembers",
    "cmtitle": CATEGORY_PREFIX_MAP,
    "cmlimit": "500",
    "format": "json",
    "formatversion": "2",
    "maxlag": "5"
}
time.sleep(3)  # 礼貌等待
response = requests.get(API_URL, headers=HEADERS, params=params)
time.sleep(1)  # 礼貌等待
print(f"请求状态码: {response.status_code}")#检查请求状态码被ban了
if(response.status_code == 200):
    title_list = jsonpath.jsonpath(json.loads(response.text), "$..title")
    
    for title in title_list:
        params = {
    "action": "query",
    "prop": "revisions",
    "titles": title,          # 要获取的页面标题
    "rvprop": "content",      # 返回页面内容（wikitext）
    "format": "json",
    "formatversion": 2,
                }
        time.sleep(1)  # 礼貌等待
        target=requests.get(API_URL,params=params)
        content=jsonpath.jsonpath(json.loads(target.text), "$..content")
        os.makedirs("./blhx", exist_ok=True)#创建目录
        with open(f"./blhx/{title}.csv", "w", encoding="utf-8") as file:
            file.write(content[0])
        
        