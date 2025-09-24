import requests
from charset_normalizer import detect
from lxml import etree
import csv
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
}

def _get(url):
    """获取网页内容"""
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        encoding = detect(response.content)['encoding']
        return response.content.decode(encoding)
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None

def _detail_get(url):
    """解析通知详情页，获取附件信息"""
    text = _get(url)
    if text:
        tree = etree.HTML(text)
        attachments = tree.xpath('//div[@class="xl_main"]/ul/li')
        attachment_info = []
        for attachment in attachments:
            try:
                # 附件名称
                attachment_name = attachment.xpath('./a/text()')[0].strip()
                # 附件下载次数
                #attachment_count = attachment.xpath('./span[starts-with(@id, "nattach")]/text()')[0].strip()
                attachment_id_full = attachment.xpath('./span/@id')[0]
                attachment_id = re.search(r'\d+', attachment_id_full).group()
                urlaj = f'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid={attachment_id}&owner=1744984858&type=wbnewsfile&randomid=nattach'
                response = requests.get(urlaj, headers=HEADERS)
                if response.status_code == 200:
                    match = re.search(r'"wbshowtimes":(\d+)', response.text)
                    attachment_count = match.group(1) if match else "0"
                else:
                    attachment_count = "-1"
                # 附件链接
                attachment_link = "https://jwch.fzu.edu.cn" + attachment.xpath('./a/@href')[0]
                attachment_info.append((attachment_name, attachment_count, attachment_link))
            except Exception as e:
                print(f"解析附件时出错: {e}")
        return attachment_info
    return None  

def save_to_csv(data, filename):
    """保存数据到 CSV 文件"""
    headers = ["通知人", "标题", "日期", "详情链接", "附件名", "附件下载次数", "附件链接"]
    with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
        print(f"数据已保存到 {filename}")

if __name__ == '__main__':
    base_url = 'https://jwch.fzu.edu.cn/jxtz'
    data = []

    for page in range(198, 185, -1):  # 从第198页到第185页
        if page == 198:
            url = base_url + ".htm"
        else:
            url = f"{base_url}/{page}.htm"

        print(f"正在爬取第 {199-page} 页: {url}")
        text = _get(url)
        if not text:
            continue

        # 解析通知列表
        tree = etree.HTML(text)
        li_list = tree.xpath('//div[@class="box-gl clearfix"]/ul/li')

        # 获取通知详情
        for li in li_list:
            try:
                # 通知详情链接
                detail_url = "https://jwch.fzu.edu.cn/" + li.xpath('./a/@href')[0]
                # 通知人
                notifier_text = li.xpath('./span[@class="doclist_time"]/following-sibling::text()')
                notifier = re.search(r'【(.*?)】', notifier_text[0].strip()).group(1) if notifier_text else "未知通知人"
                # 标题
                title = li.xpath('./a/@title')[0].strip()
                # 日期
                date = li.xpath('./span[@class="doclist_time"]/text()')[0].strip()

                # 获取附件信息
                attachments = _detail_get(detail_url)

                if attachments:
                    for attachment_name, attachment_count, attachment_link in attachments:
                        data.append([notifier, title, date, detail_url, attachment_name, attachment_count, attachment_link])
                else:
                    data.append([notifier, title, date, detail_url, "", "", ""])
            except Exception as e:
                print(f"解析通知时出错: {e}")

    save_to_csv(data, "教务通知.csv")
    print("爬取完成，数据已保存到 '教务通知.csv'")
