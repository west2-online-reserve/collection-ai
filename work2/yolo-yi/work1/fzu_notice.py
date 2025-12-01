import requests
from lxml import etree
import csv
import time
import json
import re
from urllib.parse import urljoin,urlparse, parse_qs

BASE_URL = "https://jwch.fzu.edu.cn/"
START_URL = "https://jwch.fzu.edu.cn/jxtz.htm"
PAGE_START = 206 #开始页码
PAGE_END = 175  #抓取到哪一页停止
OUTPUT_FILE = "fzu_jwc_notices.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Referer": "https://jwch.fzu.edu.cn/jxtz.htm"
}

def setup():
    #初始化CSV文件
    with open(OUTPUT_FILE, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['日期', '通知人', '标题', '详情链接', '附件名', '下载次数', '附件链接'])

def clean_text(text):
    #清洗文本：去除回车、空格、无用符号
    if not text:
        return ""
    return text.strip().replace('\r', '').replace('\n', '').replace('\t', '')

def extract_notifier(text_nodes):
    #从文本中提取【通知人】
    full_text = "".join(text_nodes)
    match = re.search(r'【(.*?)】', full_text)
    return match.group(1)

def get_download_count(wbnewsid, owner, type='wbnewsfile', randomid='nattach'):
    #附件下载次数
    #构造 API 请求地址
    api_url = f'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid={wbnewsid}&owner={owner}&type={type}&randomid={randomid}'
    try:
        response = requests.get(api_url, headers=HEADERS, timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
                if 'wbshowtimes' in data:
                    return str(data['wbshowtimes'])
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"获取次数出错: {e}")
    return '0'

def parse_detail_page(url):
    #解析详情页，提取附件名和链接
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = 'utf-8'
        html = etree.HTML(resp.text)

        attachments_data = []

        # 定位附件链接 (包含 download.jsp 的 a 标签)
        attachment_nodes = html.xpath('//a[contains(@href, "download.jsp")]')

        if not attachment_nodes:
            return None  # 无附件

        for node in attachment_nodes:
            # 提取附件名
            att_name = clean_text("".join(node.xpath('.//text()')))
            # 提取附件链接
            att_href = node.xpath('./@href')[0]
            att_full_link = urljoin(BASE_URL, att_href)

            parsed_url = urlparse(att_full_link)
            qs = parse_qs(parsed_url.query)
            # 提取参数 (注意：qs返回的是列表)
            # API 参数名是 wbnewsid，但链接里的参数名是 wbfileid
            file_id = qs.get('wbfileid', [''])[0]
            owner_id = qs.get('owner', [''])[0]
            if file_id and owner_id:
                count = get_download_count(wbnewsid=file_id, owner=owner_id)
            else:
                count = '无法解析'

            attachments_data.append({
                'name': att_name,
                'link': att_full_link,
                'count': count
            })
        return attachments_data
    except Exception as e:
        print(f"详情页解析错误: {url} - {e}")
        return None

def crawl_list_page(url):
    print(f"正在抓取列表页: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = 'utf-8'
        html = etree.HTML(resp.text)
        items = html.xpath('//ul[@class="list-gl"]/li')
        parsed_count = 0

        for item in items:
            # 提取日期
            date = item.xpath('./span[@class="doclist_time"]/text()')
            date = clean_text(date[0]) if date else ""
            # 提取链接和标题
            link_node = item.xpath('./a')
            if not link_node:
                continue
            title_attr = link_node[0].xpath('./@title')
            title = clean_text(title_attr[0]) if title_attr else clean_text(link_node[0].text)

            href = link_node[0].xpath('./@href')[0]
            full_url = urljoin(BASE_URL, href)

            # 提取通知人
            raw_text_nodes = item.xpath('./text()')
            notifier = extract_notifier(raw_text_nodes)
            # 进入详情页获取附件信息
            attachments = parse_detail_page(full_url)
            # 写入CSV
            with open(OUTPUT_FILE, mode='a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                if attachments:
                    for att in attachments:
                        # 一条通知有多个附件，就存多行
                        writer.writerow([
                            date, notifier, title, full_url,
                            att['name'], att['count'], att['link']
                        ])
                else:
                    # 无附件的情况
                    writer.writerow([date, notifier, title, full_url, "无", "", ""])
            parsed_count += 1
            time.sleep(0.2)
        return parsed_count
    except Exception as e:
        print(f"[!] 列表页错误: {e}")
        return 0

def main():
    setup()
    total_items = 0
    # 1. 抓取首页
    total_items += crawl_list_page(START_URL)
    # 2. 循环抓取分页
    for page_num in range(PAGE_START, PAGE_END, -1):
        if total_items >= 500:
            break
        page_url = urljoin(BASE_URL, f"jxtz/{page_num}.htm")
        count = crawl_list_page(page_url)
        total_items += count
        time.sleep(1)  # 翻页延时
    print(f"\n爬虫结束！共抓取 {total_items} 条数据。")
    print(f"文件已保存为: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()