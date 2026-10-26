"""
爬取福州大学教务处的通知公告，获取标题、正文、发布时间、附件信息等，并将附件下载到本地
默认情况下爬前30页共600条
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
from urllib.parse import urljoin
import csv

field = [
    "通知人",
    "标题",
    "发布时间",
    "详细链接",
    "正文",
    "附件信息",
]
BASE = Path(__file__).parent
base_url = "https://jwch.fzu.edu.cn/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36 Edg/156.0.0.0",
}
counter = 0
appendix_counter = 0
defeat_connect=[]
with open(BASE / "notice.csv", "a", encoding="utf-8", newline="") as notice_write:# 放到循环外面，只 open 一次，提高效率
    for page in range(216, 186, -1):  # 这里调爬取范围，一页20个数据，从216开始
        if page == 216:
            url = "https://jwch.fzu.edu.cn/jxtz.htm"
            print("正在爬取第1页")
        else:
            url = f"https://jwch.fzu.edu.cn/jxtz/{page}.htm"
            print(f"正在爬取第{217-page}页")
        notice_dict = dict.fromkeys(field, "无")  # 初始化字典
        # print(notice_dict)
        retry = 3# 三次重连机会
        while retry>0:
            try:
                fzu_html = requests.get(url, headers=headers,timeout=10)# 10秒超时机会
                break
            except requests.exceptions.RequestException as e:# 捕获所有网络报错
                print(f"请求失败: {e}")
                retry -= 1
                print(f"重试中...剩余尝试次数: {retry}")
                if retry == 0:
                    print("请求失败，跳过该页")
                    defeat_connect.append(url)
                    continue
            
        # 最初会403报错，根据 deepseek 说是由于反爬机制，结果搞半天是网址输错了，在jxtz后面多加了个'/'
        fzu_html.encoding = "utf-8"
        # print(fzu_html.status_code)
        soup = BeautifulSoup(fzu_html.text, "html.parser")
        ul = soup.find("ul", class_="list-gl")
        li = ul.find_all("li")
        for each in li:  # 先将所有数据提取出来
            # print(each)
            announcer = re.search(r"【(.*?)】", each.text).group(1).strip()
            notice_dict["通知人"] = announcer  # 通知人
            # print(announcer)
            # notice_dict["发布时间"]=time = (each.find("span",class_="doclist_time")).string.strip()  # 时间
            # 在外面显示的时间不太好爬，有的加了个显示颜色，不如取详细页面爬
            # print(time)
            tag_a = each.find("a")
            # print(tag_a)
            if tag_a:
                raw_url = tag_a.get("href")  # url的后半部分
                detailed_url = urljoin(base_url, raw_url).strip()
                notice_dict["详细链接"] = detailed_url  # 详细链接

                title = tag_a.get("title").strip()
                notice_dict["标题"] = title  # 标题
            retry=3
            while retry>0:
                try:
                    body_url = requests.get(detailed_url, headers=headers,timeout=10)
                    break
                except requests.exceptions.RequestException as e:
                    print(f"请求失败: {e}")
                    retry -= 1
                    print(f"重试中...剩余尝试次数: {retry}")
                    if retry == 0:
                        print("请求失败，跳过该通知")
                        defeat_connect.append(detailed_url)
                        continue
            # print(body_url)
            body_url.encoding = "utf-8"
            body_html = BeautifulSoup(body_url.text, "html.parser")
            body_text = body_html.find("div", class_="v_news_content").text.strip()
            notice_dict["正文"] = body_text  # 正文

            time = (
                body_html.find("span", class_="xl_sj_icon")
                .text.replace("发布时间：", "")
                .strip()
            )
            notice_dict["发布时间"] = time  # 发布时间
            # print(body_text)
            appendix_wrapper = body_html.find("ul", style="list-style-type:none;")
            all_appendix = appendix_wrapper.find_all("li") if appendix_wrapper else None
            if all_appendix:  # 开始对切片的循环
                # 在这里提前建立列表，到时候直接拼接成字符串输入到 csv 中
                appendix_name = []
                appendix_url = []
                appendix_info_list = []
                download_times = []
                # 其实写完这么多也有思考会不会用字典好点
                for i, appendix in enumerate(all_appendix):
                    # print(appendix)
                    # appendix_text = appendix.text.strip()
                    tag_a = appendix.find("a")
                    if tag_a:  # 下面是对tag_a操作，获取附件所有信息
                        appendix_name.append(tag_a.text.strip())
                        appendix_download_url = urljoin(base_url, tag_a.get("href")).strip()
                        appendix_url.append(appendix_download_url)
                        wbnewsid = re.search(
                            r"wbfileid=(\d+)", appendix_download_url
                        ).group(1)
                        download_times_url = (
                            "https://jwch.fzu.edu.cn/system/resource/code/news/click/"
                            f"clicktimes.jsp?wbnewsid={wbnewsid}&owner=1744984858"
                            "&type=wbnewsfile&randomid=nattach"
                        )
                        retry=3
                        while retry>0:
                            try:
                                download_times_json = requests.get(
                                    download_times_url, headers=headers, timeout=10
                                ).json()
                                break
                            except requests.exceptions.RequestException as e:
                                print(f"请求失败: {e}")
                                retry -= 1
                                print(f"重试中...剩余尝试次数: {retry}")
                                if retry == 0:
                                    print("请求失败，无法读取下载次数")
                                    defeat_connect.append(appendix_download_url)
                                    break
                        download_time = download_times_json.get("wbshowtimes", "无法读取")
                        download_times.append(download_time)
                    # 先将获取到的附件信息组合成单条信息
                    info_show = f"附件名称：{appendix_name[i]}，下载次数：{download_times[i]}，下载链接：{appendix_url[i]}"
                    appendix_info_list.append(info_show)  # 把列表信息组装起来
                    # 下面开始附件的下载
                    download_dir = BASE / "appendix"
                    download_dir.mkdir(exist_ok=True, parents=True)
                    appendix_download = requests.get(appendix_url[i], headers=headers)
                    appendix_download_file = BASE / "appendix" / f"{appendix_name[i]}"
                    with open(appendix_download_file, "wb") as appendix_write:# 本来想像通知那样只 open 一次，但是和 ai 讨论后无法实现，故只能放到循环里面
                        appendix_write.write(appendix_download.content)
                    print(f"正在下载附件：{appendix_name[i]}")
                    appendix_counter += 1
                appendix_info = "\n".join(appendix_info_list)  # 将列表信息拼接成字符串
                notice_dict["附件信息"] = appendix_info  # 附件信息
            #with open(BASE / "notice.csv", "a", encoding="utf-8", newline="") as f:#原本把写入放在这里
                writer = csv.DictWriter(notice_write, fieldnames=field)
                if notice_write.tell() == 0:
                    writer.writeheader()
                writer.writerow(notice_dict)
                # print(appendix_information)
            counter += 1
print(f"爬取结束，共爬取{counter}条数据,查看py文件旁边的{BASE / 'notice.csv'}")
print(f"已下载{appendix_counter}个附件，查看py文件旁边的{BASE / 'appendix'}文件夹")
if defeat_connect:
    print("以下链接请求失败，请检查网络或网址是否正确：")
    for url in defeat_connect:
        print(url)
