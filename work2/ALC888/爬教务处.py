import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin

cookies = {
    'JSESSIONID': '0CBEF246A1315DE3E48D06EDCC952BDE',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Referer': 'https://github.com/west2-online/learn-AI/blob/main/tasks(2025)/task1-3/task2.md',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0',
    'sec-ch-ua': '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    # 'Cookie': 'JSESSIONID=0CBEF246A1315DE3E48D06EDCC952BDE',
}
output_file = 'jwch_outputs.csv'
with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['page', 'url', 'li_text', 'a_text', 'a_href'])
    for i in range(150, 207):
        url = f'https://jwch.fzu.edu.cn/jxtz/{i}.htm'
        resp = requests.get(url, cookies=cookies, headers=headers)
        resp.encoding = resp.apparent_encoding  # 或者 resp.encoding = 'gbk'
        soup = BeautifulSoup(resp.text, 'html.parser')
        texts = soup.find_all('li')
        for li in texts:
            li_text = li.get_text(strip=True)
            if not li_text:
                li_text = repr(li)
            anchors = li.find_all('a')
            if anchors:
                for a in anchors:
                    a_text = a.get_text(strip=True)
                    href = a.get('href') or ''
                    href_abs = urljoin(url, href) if href else ''
                    writer.writerow([i, url, li_text, a_text, href_abs])
            else:
                writer.writerow([i, url, li_text, '', ''])
print(f'已将抓取结果保存到 {output_file}')
