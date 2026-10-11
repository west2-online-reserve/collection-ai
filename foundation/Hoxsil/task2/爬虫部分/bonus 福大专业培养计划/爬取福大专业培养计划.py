import requests
import os
import json
from bs4 import BeautifulSoup


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
STORAGE_FILE_PATH = os.path.join(FOLDER_PATH, "福大专业培养计划.txt")


headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'cache-control': 'max-age=0',
    'priority': 'u=0, i',
    'referer': 'https://zsks.fzu.edu.cn/info/1032/1232.htm',
    'sec-ch-ua': '"Chromium";v="154", "Microsoft Edge";v="154", "Not A(Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36 Edg/154.0.0.0',
}


def extract_major_detail(url):
    
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 200:
        print(url, ' 请求成功')
        resp.encoding = 'UTF-8-SIG'
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        
        info = {
            "学制": "",
            "学位授予门类": "",
            "专业荣誉": "",
            "专业特色": "",
            "核心课程": "",
            "升学就业情况": "",
            "升学情况": "",
            "就业情况": ""
        }

        # 遍历所有p标签，匹配加粗标题
        for p in soup.find_all("p"):
            strong_tag = p.find("strong")
            if not strong_tag:
                continue
            title_text = strong_tag.get_text(strip=True).replace("：", "")
            # 取出当前p标签全部文本，去掉加粗标题
            full_text = p.get_text(strip=True)
            # 移除标题部分，得到后面内容
            content = full_text.replace(f"{title_text}：", "").strip()
            if title_text in info:
                info[title_text] = content
        return info
    else:
        print(url, ' 请求失败')
        return "Error"


def extract_major_links(html:str):
    soup = BeautifulSoup(html, "html.parser")
    result = []
    base_url = "https://zsks.fzu.edu.cn/"
    for a_tag in soup.find_all("a"):
        href_raw = str(a_tag.get("href"))
        text = a_tag.get_text(strip=True)
        if text.endswith("专业") and text not in ("学院专业", "学院介绍"):
            # 自动拼接成完整url，字段名改为 htm
            full_link = base_url + href_raw
            result.append({
                "major_name": text[:-2],
                "detail_info": extract_major_detail(full_link)
            })
    return result

url = "https://zsks.fzu.edu.cn/xyzy.htm"

response = requests.get(url, headers=headers)
response.encoding = 'UTF-8-SIG'
print(f"状态码：{response.status_code}")
inform_list = extract_major_links(response.text)

with open(STORAGE_FILE_PATH, 'w', encoding='utf-8') as f:
    json.dump(inform_list, f, ensure_ascii=False, indent = 2)
    print("数据存储完成")