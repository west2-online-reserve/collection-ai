import requests
from lxml import etree
import csv
import re
from urllib.parse import urljoin

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[\r\n\t]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_clicktimes(script_text):
    """通过API获取下载次数"""
    try:
        newsid = re.search(r"\(([0-9]{2,}),", script_text).group(1)
        owner = re.search(r",([0-9]{2,}),", script_text).group(1)
        url = root_url + "system/resource/code/news/click/clicktimes.jsp" + \
            f"?wbnewsid={newsid}&owner={owner}&type=wbnewsfile&randomid=nattach"
        download_json = requests.get(url, headers=headers).json()
        return download_json["wbshowtimes"]
    except Exception:
        return "0"

def get_download_count(node):
    """获取附件下载次数"""
    parent = node.getparent()
    while parent is not None:
        script_nodes = parent.xpath('.//script[contains(text(), "getClickTimes")]/text()')
        if script_nodes:
            return get_clicktimes(script_nodes[0])
        parent = parent.getparent()
    return "0"

# 存储所有通知信息
informs_list = []
max_attach = 0  # 记录最大附件数
urls = []
for i in range(206, 176, -1):
    url = "https://jwch.fzu.edu.cn/jxtz/{}.htm".format(i)
    urls.append(url)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
notices_count = 0
root_url = "https://jwch.fzu.edu.cn/"

for url in urls:
    response = requests.get(url, headers=headers, timeout=10)
    response.encoding = 'utf-8'
    content = response.text
    html = etree.HTML(content)
    lis = html.xpath('//ul[@class="list-gl"]/li')
    
    for li in lis:
        try:
            # 提取基本信息
            date_nodes = li.xpath('.//span[@class="doclist_time"]/text()')
            date = clean_text(date_nodes[0]) if date_nodes else ""
            title_nodes = li.xpath('./a[@title]')
            if len(title_nodes) == 0:
                continue
            
            title = clean_text(title_nodes[0].get('title', ''))
            relative_link = title_nodes[0].get('href', '')
            if relative_link.startswith('../'):
                relative_link = relative_link[3:]
            link = urljoin(root_url, relative_link)
            
            text_content = ''.join(li.itertext())
            publisher = ""
            if "【" in text_content and "】" in text_content:
                start = text_content.find("【") + 1
                end = text_content.find("】")
                publisher = text_content[start:end] if start < end else ""

            # 获取附件信息
            attachments = []
            detail_response = requests.get(link, headers=headers, timeout=10)
            detail_response.encoding = 'utf-8'
            detail_html = etree.HTML(detail_response.text)
            
            attachment_nodes = detail_html.xpath('//a[contains(@href, "download") or contains(@href, "attach") or contains(@href, ".pdf") or contains(@href, ".doc") or contains(@href, ".docx") or contains(@href, ".xls") or contains(@href, ".xlsx") or contains(@href, ".zip") or contains(@href, ".rar")]')
            
            # 构建数据字典
            inform = {
                '通知人': publisher,
                '标题': title,
                '日期': date,
                '详情链接': link,
            }
            
            # 处理附件信息
            for i, node in enumerate(attachment_nodes):
                attachment_name = clean_text(node.xpath('string()'))
                attachment_link = node.get('href', '')

                if attachment_link and attachment_name.strip():
                    full_attachment_link = urljoin(link, attachment_link)
                    download_count = get_download_count(node)
                    # 使用动态字段名
                    inform.update({
                        f'下载量{i+1}': download_count,
                        f'附件{i+1}': attachment_name,
                        f'附件链接{i+1}': full_attachment_link,
                    })
            # 更新最大附件数
            if len(attachment_nodes) > max_attach:
                max_attach = len(attachment_nodes)
            
            informs_list.append(inform)
            notices_count += 1
            print(f"已爬取 {notices_count} 条通知: {title}")
            
        except Exception as e:
            print(f"解析通知项失败: {e}")
            continue
with open('FZU_data.csv', 'w', encoding='utf-8-sig', newline='') as f:
    # 构建表头 - 确保与字典键完全匹配
    header = ['通知人', '标题', '日期', '详情链接']
    for i in range(max_attach):
        header.extend([ f'下载量{i+1}',f'附件{i+1}', f'附件链接{i+1}'])
    
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    
    # 逐行写入并验证
    for i, inform in enumerate(informs_list):
        # 确保字典包含所有表头字段
        row = {}
        for field in header:
            row[field] = inform.get(field, '')
        writer.writerow(row)


print(f"爬取完成！共爬取 {notices_count} 条通知")
print(f"数据已保存到 福大教务处通知.csv")