"""API 调用模块：处理与远程 API 的交互，如获取下载次数等"""

import requests
from urllib.parse import urljoin, urlparse, parse_qs
import os

def get_click_times(fileid: str, owner: str, base_url: str = 'https://jwch.fzu.edu.cn/',
                   headers: dict = None) -> str:
    """调用站点的计数接口获取附件下载次数
    
    Args:
        fileid: 文件ID
        owner: 所有者ID
        base_url: 基础URL
        headers: 请求头
        
    Returns:
        下载次数字符串，失败返回空字符串
    """
    if not fileid or not owner:
        return ''
        
    try:
        api_url = urljoin(base_url, '/system/resource/code/news/click/clicktimes.jsp')
        params = {
            'wbnewsid': str(fileid),
            'owner': str(owner),
            'type': 'wbnewsfile',
            'randomid': 'nattach'
        }
        resp = requests.get(api_url, headers=headers or {}, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get('wbshowtimes', '') or '')
    except Exception:
        return ''

def extract_file_info(url: str) -> tuple:
    """从附件URL中提取文件ID和owner
    
    Args:
        url: 附件完整URL
    
    Returns:
        (fileid, owner) 元组
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    
    # 优先使用常见的文件 id 参数
    fileid = ''
    for key in ('wbfileid', 'fileid', 'id', 'attachid', 'aid', 'newsid', 'wbnewsid'):
        if key in query and query[key]:
            fileid = query[key][0]
            break
            
    # 获取 owner 参数
    owner = query.get('owner', [''])[0]
    
    return fileid, owner

def update_attachment_download_count(attachment: dict, base_url: str, headers: dict = None) -> dict:
    """更新附件的下载次数
    
    Args:
        attachment: 附件字典
        base_url: 基础URL
        headers: 请求头
    
    Returns:
        更新后的附件字典
    """
    if not attachment.get('url'):
        return attachment
        
    fileid, owner = extract_file_info(attachment['url'])
    
    if fileid and owner:
        count = get_click_times(fileid, owner, base_url, headers)
        if count:
            attachment['download_count'] = count
        attachment['link_code'] = fileid
            
    return attachment