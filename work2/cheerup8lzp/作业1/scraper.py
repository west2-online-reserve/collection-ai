"""爬虫核心模块：实现教务通知的抓取功能"""

import requests
from bs4 import BeautifulSoup
import time
import os
import json
from urllib.parse import urljoin, urlparse, parse_qs
from typing import List, Dict, Optional
from pathlib import Path

from parsers import parse_notice_list, parse_attachments
from api import update_attachment_download_count
from utils import ensure_dir, safe_filename, download_file

from parsers import parse_notice_list, parse_attachments
from api import update_attachment_download_count
from utils import ensure_dir, safe_filename
from pathlib import Path

class FZUNoticeScraper:
    """福大教务通知爬虫类"""
    
    def __init__(self, base_url: str = 'https://jwch.fzu.edu.cn/'):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
        }
        
    def scrape_notice_list(self, target_count: int = 500, start_page: int = 205,
                          sleep_sec: float = 0.8) -> List[Dict]:
        """抓取通知列表
        
        Args:
            target_count: 目标抓取数量
            start_page: 起始页码
            sleep_sec: 请求间隔
            
        Returns:
            通知列表
        """
        results = []
        seen_links = set()
        
        # 首页
        first_url = urljoin(self.base_url, 'jxtz.htm')
        try:
            r = requests.get(first_url, headers=self.headers, timeout=15)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, 'html.parser')
            items = parse_notice_list(soup, self.base_url)
            for it in items:
                if it['详情链接'] not in seen_links:
                    results.append(it)
                    seen_links.add(it['详情链接'])
            print(f"已抓取首页，累计 {len(results)} 条")
        except requests.RequestException as e:
            print(f"请求首页时发生错误: {e}")
            
        # 后续页面
        for p in range(start_page, 0, -1):
            if len(results) >= target_count:
                break
                
            page_url = urljoin(self.base_url, f'jxtz/{p}.htm')
            try:
                time.sleep(sleep_sec)
                r = requests.get(page_url, headers=self.headers, timeout=15)
                r.raise_for_status()
                r.encoding = r.apparent_encoding
                soup = BeautifulSoup(r.text, 'html.parser')
                items = parse_notice_list(soup, self.base_url)
                
                added = 0
                for it in items:
                    if it['详情链接'] and it['详情链接'] not in seen_links:
                        results.append(it)
                        seen_links.add(it['详情链接'])
                        added += 1
                        
                print(f"抓取 {p}.htm，新增 {added} 条，累计 {len(results)} 条")
            except requests.RequestException as e:
                print(f"请求 {page_url} 时发生错误: {e}")
                continue
                
        return results
        
    def fetch_notice_detail(self, notice: Dict, sleep_sec: float = 0.6) -> Dict:
        """获取通知详情
        
        Args:
            notice: 通知字典
            sleep_sec: 请求间隔
            
        Returns:
            更新后的通知字典
        """
        detail_url = notice.get('详情链接')
        if not detail_url:
            notice['详情HTML'] = ''
            notice['附件'] = []
            return notice
            
        time.sleep(sleep_sec)
        try:
            r = requests.get(detail_url, headers=self.headers, timeout=20)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # 解析附件
            attachments = parse_attachments(soup, self.base_url)
            
            # 获取附件下载次数
            updated_attachments = []
            for att in attachments:
                updated_att = update_attachment_download_count(att, self.base_url, self.headers)
                updated_attachments.append(updated_att)
                
            notice['详情HTML'] = r.text
            notice['附件'] = updated_attachments
            
        except requests.RequestException:
            notice['详情HTML'] = ''
            notice['附件'] = []
            
        return notice
        
    def download_attachments(self, results: List[Dict], save_dir: str) -> None:
        """下载所有通知的附件
        
        Args:
            results: 通知列表
            save_dir: 附件保存目录
        """
        # 创建基础目录
        ensure_dir(save_dir)
        
        total_files = sum(len(notice.get('附件', [])) for notice in results)
        downloaded = 0
        
        for notice in results:
            # 为每个通知创建子文件夹
            notice_dir = os.path.join(save_dir, safe_filename(notice['标题'][:50]))
            
            for attachment in notice.get('附件', []):
                downloaded += 1
                print(f"\r正在下载附件 [{downloaded}/{total_files}]", end='')
                
                url = attachment.get('url')
                if not url:
                    continue
                
                filename = attachment.get('name')
                saved_path = download_file(url, notice_dir, filename, self.headers)
                
                if saved_path:
                    attachment['saved_path'] = saved_path
                    
        print(f"\n共下载了 {downloaded} 个附件到 {save_dir}")

    def save_results(self, results: List[Dict], json_file: str, csv_file: str) -> None:
        """保存结果到JSON和CSV
        
        Args:
            results: 通知列表
            json_file: JSON文件路径
            csv_file: CSV文件路径
        """
        # 保存JSON
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已将包含详情的结果保存到 {json_file}")
        except Exception as e:
            print(f"保存 JSON 时出错: {e}")
            
        # 保存CSV
        try:
            import csv
            fieldnames = ['通知人', '标题', '日期', '详情链接', '详情HTML', '附件']
            with open(csv_file, 'w', encoding='utf-8-sig', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                for it in results:
                    row = {
                        '通知人': it.get('通知人', ''),
                        '标题': it.get('标题', ''),
                        '日期': it.get('日期', ''),
                        '详情链接': it.get('详情链接', ''),
                        '详情HTML': it.get('详情HTML', ''),
                        '附件': json.dumps(it.get('附件', []), ensure_ascii=False)
                    }
                    writer.writerow(row)
            print(f"已将所有数据保存为 CSV: {csv_file}")
        except Exception as e:
            print(f"保存 CSV 时出错: {e}")
            
    def print_sample(self, results: List[Dict], sample_n: int = 20) -> None:
        """打印样例数据
        
        Args:
            results: 通知列表
            sample_n: 打印条数
        """
        print(f"\n--- 爬取结果示例（前 {sample_n} 条 / 共 {len(results)} 条）---")
        for i, item in enumerate(results[:sample_n]):
            print(f"--- 第 {i+1} 条 ---")
            print(f"通知人: {item.get('通知人','')}")
            print(f"标题: {item.get('标题','')}")
            print(f"日期: {item.get('日期','')}")
            print(f"详情链接: {item.get('详情链接','')}")
            atts = item.get('附件', [])
            print(f"附件数量: {len(atts)}")
            for a in atts:
                print(f" - 名称: {a.get('name','')}")
                print(f"   链接: {a.get('url','')}")
                print(f"   下载次数: {a.get('download_count','')}")
                print(f"   链接码: {a.get('link_code','')}")
            print('-' * 20)