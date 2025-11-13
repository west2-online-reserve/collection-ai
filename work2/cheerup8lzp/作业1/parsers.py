"""页面解析模块：负责解析列表页和详情页的 HTML 内容"""

import re
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def parse_notice_list(soup: BeautifulSoup, base_url: str) -> list:
    """从列表页面解析通知条目
    
    Args:
        soup: BeautifulSoup 对象
        base_url: 基础 URL（用于构建完整链接）
    
    Returns:
        解析出的通知列表，每个通知包含: 通知人、标题、日期、详情链接
    """
    # 日期正则
    date_regex = re.compile(r'\d{4}-\d{2}-\d{2}')
    results = []

    # 遍历所有 span 查找日期节点
    all_spans = soup.find_all('span')
    date_elements = [s for s in all_spans if s.text and date_regex.search(s.text.strip())]

    for date_span in date_elements:
        parent = date_span.parent
        link_tag = parent.find('a') if parent else None

        if not link_tag or not link_tag.text:
            continue

        href = link_tag.get('href')
        detail_link = urljoin(base_url, href) if href else ''
        date = date_span.text.strip()
        raw_text = link_tag.text
        cleaned_raw_text = raw_text.replace('\r', '').replace('\n', '').strip()

        # 提取通知人和标题
        match = re.search(r'【(.*?)】(.*)', cleaned_raw_text)
        if match:
            notifier = match.group(1).strip()
            title_part = match.group(2).strip()
            title = title_part.replace('(', '').replace(')', '').replace('（', '').replace('）', '').strip()
        else:
            title = cleaned_raw_text.replace('(', '').replace(')', '').replace('（', '').replace('）', '').strip()
            parent_text = parent.get_text(separator=' ', strip=True) if parent is not None else ''
            parent_match = re.search(r'【(.*?)】', parent_text)
            notifier = parent_match.group(1).strip() if parent_match else 'N/A'

        results.append({
            "通知人": notifier,
            "标题": title,
            "日期": date,
            "详情链接": detail_link
        })

    return results

def parse_attachments(soup: BeautifulSoup, base_url: str) -> list:
    """从详情页解析附件信息
    
    Args:
        soup: BeautifulSoup 对象
        base_url: 基础 URL
        
    Returns:
        附件列表，每个附件包含: 名称、URL、下载次数（初始为空）、link_code
    """
    attachments = []
    exts = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.txt', '.jpg', '.png')
    
    for a in soup.find_all('a', href=True):
        href = a.get('href').strip()
        lower = href.lower()
        is_attach = lower.endswith(exts) or '附件' in a.get_text() or 'download' in lower
        if not is_attach:
            continue

        attach_url = urljoin(base_url, href)
        name = a.get_text(strip=True) or os.path.basename(urlparse(attach_url).path)

        # 基础信息
        attachment = {
            'name': name,
            'url': attach_url,
            'download_count': '',  # 初始为空，后续由 api 模块填充
            'link_code': '',  # 初始为空，后续处理
            'saved_path': ''
        }
        
        attachments.append(attachment)

    return attachments