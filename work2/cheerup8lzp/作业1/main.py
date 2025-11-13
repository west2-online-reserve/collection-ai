"""主程序入口：执行教务通知爬取任务"""

import os
from typing import Optional, Dict
from datetime import datetime
from scraper import FZUNoticeScraper
from utils import ensure_dir

def main():
    # 初始化爬虫
    scraper = FZUNoticeScraper()
    
    # 设置输出目录和文件
    output_dir = 'output'
    ensure_dir(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = os.path.join(output_dir, f'notices_{timestamp}.json')
    csv_file = os.path.join(output_dir, f'notices_{timestamp}.csv')
    
    # 抓取通知列表
    print("开始抓取通知列表...")
    results = scraper.scrape_notice_list(target_count=500, start_page=205)
    print(f"\n共抓取到 {len(results)} 条通知")
    
    # 获取通知详情
    print("\n开始获取通知详情和附件信息...")
    for i, notice in enumerate(results, 1):
        updated = scraper.fetch_notice_detail(notice)
        print(f"[{i}/{len(results)}] 已获取通知详情：{updated.get('标题', '')[:30]}...")
        
    # 下载附件
    attachments_dir = os.path.join(output_dir, f'attachments_{timestamp}')
    print("\n开始下载附件...")
    scraper.download_attachments(results, attachments_dir)
    
    # 保存结果
    scraper.save_results(results, json_file, csv_file)
    
    # 打印样例数据
    scraper.print_sample(results)
    
if __name__ == '__main__':
    main()