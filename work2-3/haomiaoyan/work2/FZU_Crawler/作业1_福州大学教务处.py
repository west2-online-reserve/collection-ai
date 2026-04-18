# 1.拿到主页面(福州大学教务处)的源代码，然后提取到子页面(教学通知)的链接地址，href
# 2.通过href拿到子页面(信息)的内容
# 3.在子页面中提取数据

import requests
import re
from lxml import etree
import csv
from urllib.parse import urljoin  # 导入URL拼接工具
import time


# 在教务处主界面中提取到子页面(教学通知)的url
def get_sub_page(sub_url):
    sub_resp = requests.get(sub_url, headers=header)
    sub_resp.encoding = "utf-8"
    sub_tree = etree.HTML(sub_resp.text)

    """/html/body/div/div[3]/div[2]/div[2]/div[1]/div[2]/div[1]/div/a"""  # 绝对路径 => url_teaching
    # 这里运用的相对路径进行查找
    url_teaching = sub_tree.xpath("//div[@class=\"impor-right fr\"]/div[1]/div/a/@href")[0]  # 从教务处中提取教学通知的url（子页面）
    # 应该得到的url_teaching => 'jxtz.htm'

    # 现在要进行拼接url, 将提取到的url_teaching与url进行拼接，就可以得到子页面的url
    return sub_url + url_teaching


# 从JS脚本中提取下载次数的参数（因下载次数是动态加载，暂提取参数）
def extract_download_params(script_text):
    pattern = r'getClickTimes\((\d+),(\d+),\"(.*?)\",\"(.*?)\"\)'
    match = re.search(pattern, script_text)
    if match:
        return f"wbfileid={match.group(1)},owner={match.group(2)}"
    return "未知参数"


# 在教学通知中提取到信息的url,每一页有20条信息,并将它们保存到列表中,方便后续使用
def get_sub_message(sub_url, main_url):
    # 用于存储所有有效的real_url
    url_list = []

    sub_resp = requests.get(sub_url, headers=header)
    sub_resp.encoding = "utf-8"
    sub_tree = etree.HTML(sub_resp.text)

    # 这里运用的相对路径进行查找
    url_message = sub_tree.xpath("//ul[@class=\"list-gl\"]/li")  # 从教学通知里提取消息

    # 拿到教学通知一个页面中的信息的url
    for li in url_message:
        # 获取href列表，strip()去除首尾空白
        href_list = li.xpath("./a/@href")
        if not href_list:  # 无匹配的href，跳过当前li
            continue
        href = href_list[0].strip()

        # 处理路径拼接，避免重复的/
        if href.startswith("/"):
            real_url = main_url + href
        else:
            real_url = f"{main_url}/{href}"

        url_list.append(real_url)
        # print(f"已收集URL：{real_url}")  # 打印日志，方便调试

    return url_list


# 提取信息中需要的文本，标题，日期等, 并存在文件(csv)中
def get_message(sub_url):
    if not sub_url:  # 若传入空列表
        print("为空列表!")
        return
    for i in sub_url:
        i_resp = requests.get(i, headers=header)
        i_resp.encoding = "utf-8"
        # i_html = i_resp.text
        i_tree = etree.HTML(i_resp.text)
        # 通知人
        i_people_list = i_tree.xpath("//p[@class=\"w-main-dh-text\"]/a[3]/text()")
        i_people = i_people_list[0].strip() if i_people_list else ""
        # 通知时间
        i_time_list = i_tree.xpath("//div[@class=\"fl xl_sj\"]/span[1]/text()")
        i_time = i_time_list[0].strip() if i_time_list else ""
        # 标题
        i_title_list = i_tree.xpath("//div[@class=\"xl_tit\"]/h4/text()")
        i_title = i_title_list[0].strip() if i_title_list else ""
        # 信息文本
        i_html = i_resp.text
        i_html = i_html.replace("\n", "").replace("\r", "").replace("\t", " ")
        i_html = i_html.replace("\ufeff", "")
        # 提取附件列表
        i_attachment_list = i_tree.xpath("//ul[contains(@style, 'list-style-type:none;')]/li")
        has_attach = "有" if i_attachment_list else "无"
        attach_info = []  # 存储附件名和链接
        download_params = []  # 存储下载次数参数

        # 遍历附件，提取信息
        for i_attachment in i_attachment_list:
            # 提取附件名
            attach_name = i_attachment.xpath("./a/text()")[0] \
                if i_attachment.xpath("./a/text()") else "未知附件"
            # 提取附件链接并拼接为完整URL
            attach_href = i_attachment.xpath("./a/@href")[0].strip() if i_attachment.xpath("./a/@href") else ""
            attach_url = urljoin(BASE_URL, attach_href)
            attach_info.append(f"{attach_name}({attach_url})")

            # 提取下载次数的JS脚本，解析参数
            script_text = i_attachment.xpath("./span/script/text()")[0].strip() if i_attachment.xpath(
                "./span/script/text()") else ""
            param = extract_download_params(script_text)
            download_params.append(param)

        # 拼接附件信息和下载参数
        attach_info_str = "; ".join(attach_info) if attach_info else "无"
        download_params_str = "; ".join(download_params) if download_params else "无"

        # 写入数据CSV（包含附件信息）
        csvwriter.writerow([i_people, i_title, i_time, i, has_attach, attach_info_str, download_params_str])
        # 写入HTML CSV
        csvwriter_html.writerow([i, i_html])


if __name__ == '__main__':

    # 存储数据
    f_data = open("数据.csv", mode="w", encoding="utf-8", newline="")
    csvwriter = csv.writer(f_data)
    csvwriter.writerow(["发布部门", "标题", "发布时间", "通知URL", "是否有附件", "附件信息", "下载次数参数"])

    # 存储html
    f_html = open("html.csv", mode="w", encoding="utf-8", newline="")
    csvwriter_html = csv.writer(f_html)

    header = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
    }

    url = "https://jwch.fzu.edu.cn/"  # 福州大学教务处url
    BASE_URL = "https://jwch.fzu.edu.cn/"  # 基础URL，用于拼接附件链接

    sub_page_href = get_sub_page(url)  # 教学通知的url
    print(sub_page_href)

    for num in range(200, 149, -1):
        time.sleep(0.5)
        page_href = f"{url}jxtz/{num}.htm"
        real_url = get_sub_message(page_href, url)
        get_message(real_url)

    # get_sub_message(sub_page_href, url)  # 教学通知里的20条信息的url，以列表存储

    # get_message(get_sub_message(sub_page_href, url))

    # 关闭文件
    f_data.close()
    f_html.close()

    print("over!!!")

# 对于本作业，当爬取的数量很大时， 可以运用线程池来优化效率，不过由于本次作业写得很急(ddl快到了)， 所以就没优化
