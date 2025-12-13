import requests
import pathlib
import json
import csv
import re
import urllib.parse
from bs4 import BeautifulSoup

if __name__ == "__main__":
    base_url='https://jwch.fzu.edu.cn/'
    headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0'}
    out_path=pathlib.Path(__file__).parent /"福大教务处通知数据"
    out_path.mkdir(exist_ok=True)
    csv_path=out_path /"通知数据.csv"
    file_out_path=out_path /"通知附件"
    file_out_path.mkdir(exist_ok=True)
    records = []
    file_records = []

    for start_num in range(206,180,-1):
        get_url=f'https://jwch.fzu.edu.cn/jxtz/{start_num}.htm'
        response=requests.get(url=get_url,headers=headers)
        response.encoding ='utf-8'
        html_str=response.text
        soup = BeautifulSoup(html_str, "html.parser")
        all_date=soup.find_all('span', attrs={'class':'doclist_time'})
        for date in all_date:
            date_string=date.get_text(strip=True)
            li=date.parent
            text=li.get_text(" ",strip=True)
            print(text)
            m=re.search(r"【(.*?)】",text)
            notifier=m.group(1).strip()
            a=li.find('a')
            title=a.get("title")
            half_url=a.get("href")
            full_url=urllib.parse.urljoin(base_url,half_url)
            records.append({'通知人': notifier, '标题': title, '日期': date_string, '链接': full_url})

    with csv_path.open("w",encoding ="utf-8",newline='') as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=['通知人','标题','日期','链接'])
        writer.writeheader()
        writer.writerows(records)
        print(f"CSV 已保存 -> {csv_path}")
    csvfile.close()

    with open(csv_path,'r',encoding='utf-8') as csvfile:
        reader=csv.DictReader(csvfile)
        records=list(reader)
    for record in records:
        html_url=record['链接']
        file_notifier=record['通知人']
        print(f"正在处理通知：{record['标题']} -> {html_url}")
        response=requests.get(url=html_url,headers=headers)
        response.encoding ='utf-8'
        file_html_str=response.text
        if 'download.jsp' not in file_html_str:
            print("无附件，跳过")
            continue
        else:
            file_soup = BeautifulSoup(file_html_str,'html.parser')
            # 在详情页里寻找所有包含 download.jsp 的链接
            attach_links = file_soup.select('a[href*="download.jsp"]')
            if not attach_links:
                print("标记到 download.jsp 字样但未解析到具体 <a>")
                continue
            for a_tag in attach_links:
                file_url = urllib.parse.urljoin(base_url, a_tag['href'])
                file_name = a_tag.get_text(strip=True) or '附件'
                prased=urllib.parse.urlparse(file_url)
                qs=urllib.parse.parse_qs(prased.query)
                owener=qs.get('owner')
                wbfileid=qs.get('wbfileid')
                file_download_params={'wbnewsid':wbfileid,'owner':owener,'type':'wbnewsfile','randomid':'nattach'}
                requests_url=f"https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp"
                file_download_response=requests.get(url=requests_url,params=file_download_params,headers=headers)
                file_download_info=file_download_response.text
                wbshowtimes=None
                file_download_data=json.loads(file_download_info)
                wbshowtimes=file_download_data.get('wbshowtimes')
                print('发现附件:', file_name, '->', file_url)
                try:
                    file_response = requests.get(url=file_url, headers=headers, timeout=20)
                except Exception as e:
                    print(f"下载失败: {file_url} -> {e}")
                    continue
                file_path = file_out_path / file_name
                # 若重名追加序号
                if file_path.exists():
                    stem, suffix = file_path.stem, file_path.suffix
                    idx = 1
                    while True:
                        new_path = file_out_path / f"{stem}_{idx}{suffix}"
                        if not new_path.exists():
                            file_path = new_path
                            break
                        idx += 1
                with file_path.open('wb') as fp:
                    fp.write(file_response.content)
                print(f"已下载附件 -> {file_path}")
                # 记录附加信息：通知人、附件名、下载次数
                file_records.append({'通知人': file_notifier, '附件名': file_name, '下载次数': wbshowtimes})
    file_csv_path=out_path /"通知附件数据.csv"
    with file_csv_path.open("w",encoding ="utf-8",newline='') as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=['通知人','附件名','下载次数'])
        writer.writeheader()
        writer.writerows(file_records)
        print(f"附件数据 CSV 已保存 -> {file_csv_path}")
            
        