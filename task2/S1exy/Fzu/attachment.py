import requests
import get_download_numbers

"""
对于附件的处理 又调用了一个专门从后端找 下载次数的 模块 为 get_download_numbers
"""
from lxml import etree
def get_html(url):
    """
    # 福州大学教务处 页面获取处理
    :param url: 获取的网页（可以是每一个单独的主页）
    :return: 返回的数据为福大的主页html （etree类型）
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
    }
    response = requests.get(url, headers=headers)
    html_etree = etree.HTML(response.text)
    return html_etree

def get_istrue(html_etree):
    file_download = []
    file_name2 = []

    name = html_etree.xpath("/html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li")
    if len(name) == 0:
        return len(name),file_name2, file_download
    for file_name in name:
        file_name1 = file_name.xpath("./a/text()")[0]
        # file_name2.append(file_name1.encode('iso-8859-1').decode('utf-8'))
        try:
            # 尝试执行的代码
            file_name2.append(file_name1.encode('iso-8859-1').decode('utf-8'))
        except UnicodeEncodeError:
            file_name2.append(file_name1.encode('utf-8').decode('utf-8'))
            print("党政文件")

        file_download_path = file_name.xpath("./span/@id")[0][7:]

        file_download_path = "https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid=" + file_download_path + "&owner=1744984858&type=wbnewsfile&randomid=nattach"

        file_download.append(get_download_numbers.get_download_numbers(file_download_path))
    return len(name),file_name2, file_download


get_istrue(get_html("https://jwch.fzu.edu.cn/info/1037/13647.htm"))