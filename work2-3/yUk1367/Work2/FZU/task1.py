# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup

def deal_getClick(s):
    id1 = s.split(',')[0][14:]
    id2 = s.split(',')[1]
    r = requests.get(url=f"https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid={id1}&owner={id2}&type=wbnewsfile&randomid=nattach")
    r.encoding = 'utf-8'
    import ast
    d = ast.literal_eval(r.text)
    return d['wbshowtimes']


headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"}

for jxtz_id in [''] + [f'/{i}' for i in range(206,157,-1)]:
    jxtz = requests.get(f"https://jwch.fzu.edu.cn/jxtz{jxtz_id}.htm", headers = headers)
    jxtz.encoding = 'utf-8'
    soup = BeautifulSoup(jxtz.text, "lxml")

    all_lis = [li for li in soup.find_all('li') if '20' in li.get_text()]

    for single_li in all_lis:
        try:
            single_time = single_li.find('span').get_text(strip=True)
            single_publisher = single_li.find_all(string=True, recursive=False)[1].strip()[1:-1]
            single_information = single_li.find('a')
            if not single_information:
                continue
            single_href = 'https://jwch.fzu.edu.cn/' + single_information["href"].strip()
            single_title = single_information["title"]
            content = requests.get(single_href, headers = headers, timeout=3)
            content.encoding = 'utf-8'
            content_soup = BeautifulSoup(content.text, "lxml")
            attachment_lis = [li for li in content_soup.find_all('li') if '附件' in li.get_text()]
            attachment_diclist = []
            for attachment_li in attachment_lis:
                a = attachment_li.find("a")
                dic = {}
                span = attachment_li.find("span")
                if a and a.get("href"):
                    
                    dic["counts"] = deal_getClick(span.string)
                    dic["href"] = a["href"]
                    dic["name"] = a.get_text()

                    attachment_diclist.append(dic)


            print(f"标题：{single_title} \n日期：{single_time} \n通知人：{single_publisher} \n详细链接：{single_href}")
            if attachment_diclist:
                for att in attachment_diclist:
                    print(f"    {att['name']} 下载次数：{att['counts']} \n      链接码：{'https://jwch.fzu.edu.cn/' + att['href'][3:]}")
            print("\n")
        except Exception as e:
            print(f"出错：{e}")
            continue






