import requests
import re
import os
import pandas as pd

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}


def normal_get_id(li, url):
    response = requests.get(url=url, headers=headers).text
    pattern = r'info/\d+/(\d+)\.htm'
    li.extend(re.findall(pattern, response))


def add_normal_url(agent, num):
    return 'https://jwch.fzu.edu.cn/info/' + agent + '/' + num + '.htm'


def get_id(agent, num_list, url, num, ch):
    normal_get_id(num_list, url)
    num -= 1
    while num >= 1:
        current_url = 'https://jwch.fzu.edu.cn/jxtz/' + ch + '/' + str(num) + '.htm'
        response = requests.get(url=current_url, headers=headers).text
        pattern = r'info/\d+/(\d+)\.htm'
        num_list.extend(re.findall(pattern, response))
        num -= 1


def normal_get_one_resorces(agent, url, resource_list):
    dic = {}

    response = requests.get(url=url, headers=headers)
    response.encoding = 'utf-8'

    dic['agent'] = agent
    dic['title'] = re.search(r'<title>(.*?)</title>', response.text).group(1)
    data_resorce = (re.search(r'发布时间：(.*?)</span>', response.text))
    if data_resorce:
        dic['date'] = data_resorce.group(1)
    dic['url'] = url

    #这里检查网页源码有没有附件之后，使用阿贾克斯请求来获取附件的个数和下载次数
    dic['attachment'] = {}
    dic['attachment']['url'] = []
    dic['attachment']['name'] = []
    dic['attachment']['webfile_id'] = []
    dic['attachment']['times'] = []

    pattern = r'info/\d+/(\d+)\.htm'

    match_response = re.findall(
        r'a href="(/system/_content/download.jsp\?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=(.*?))" target="_blank">(.*?)</a>',
        response.text)

    for url, webfile_id, name in match_response:
        dic['attachment']['url'].append('https://jwch.fzu.edu.cn' + url)
        dic['attachment']['webfile_id'].append(webfile_id)
        dic['attachment']['name'].append(name)

    for id in dic['attachment']['webfile_id']:
        file_url = 'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid=' + id + '&owner=1744984858&type=wbnewsfile&randomid=nattach'
        response_id = requests.get(url=file_url, headers=headers).text
        times_match = re.search(r'{"wbshowtimes":(\d+),', response_id).group(1)
        dic['attachment']['times'].append(times_match)

    resource_list.append(dic)

    # 提取通知信息中的“通知人”(如：质量办、计划科)、标题、日期、详情链接。
    # 可能存在“附件”，提取附件名，附件下载次数，附件链接吗，有能力请尽可能将附件爬取下来。


def normal_get_all_resorces(resource_list, num_list, agent):
    for num in num_list:
        url = add_normal_url(agent, num)
        normal_get_one_resorces(agent, url, resource_list)


def other_get_all_resorces(resource_list, num_list, agent):
    for num in num_list:
        url = 'https://jwch.fzu.edu.cn/content.jsp?urltype=news.NewsContentUrl&wbtreeid=1042&wbnewsid=' + num
        normal_get_one_resorces(agent, url, resource_list)


if __name__ == '__main__':
    zhk_agent = '1035'
    jxyx_agent = '1036'
    jyjg_agent = '1037'
    jhk_agent = '1038'
    sjk_agent = '1039'
    zlb_agent = '1040'
    tpxqglk_agent = '1152'

    # 这里还差一个教材中心

    jczx_agent = '1042'

    zhk_num_list = []
    jxyx_num_list = []
    jyjg_num_list = []
    jhk_num_list = []
    sjk_num_list = []
    zlb_num_list = []
    tpxqglk_num_list = []

    jczx_num_list = []

    zhk_url = 'https://jwch.fzu.edu.cn/jxtz/zhk.htm'
    jxyx_url = 'https://jwch.fzu.edu.cn/jxtz/jxyx.htm'
    jyjg_url = 'https://jwch.fzu.edu.cn/jxtz/jyjg.htm'
    jhk_url = 'https://jwch.fzu.edu.cn/jxtz/jhk.htm'
    sjk_url = 'https://jwch.fzu.edu.cn/jxtz/sjk.htm'
    zlb_url = 'https://jwch.fzu.edu.cn/jxtz/zlb.htm'
    tpxqglk_url = 'https://jwch.fzu.edu.cn/jxtz/tpxqglk.htm'

    jczx_url = 'https://jwch.fzu.edu.cn/jxtz/jczx.htm'

    zhk_resource_list = []  # 封装一个列表，列表里的元素是字典
    jxyx_resource_list = []
    jyjg_resource_list = []
    jhk_resource_list = []
    sjk_resource_list = []
    zlb_resource_list = []
    tpxqglk_resource_list = []

    jczx_resource_list = []

    # 获取id
    get_id(zhk_agent, zhk_num_list, zhk_url, 20, 'zhk')
    get_id(jxyx_agent, jxyx_num_list, jxyx_url, 80, 'jxyx')
    get_id(jyjg_agent, jyjg_num_list, jyjg_url, 22, 'jyjg')
    get_id(jhk_agent, jhk_num_list, jhk_url, 10, 'jhk')
    get_id(sjk_agent, sjk_num_list, sjk_url, 45, 'sjk')
    get_id(zlb_agent, zlb_num_list, zlb_url, 17, 'zlb')
    get_id(tpxqglk_agent, tpxqglk_num_list, tpxqglk_url, 4, 'tpxqglk')

    get_id(jczx_agent, jczx_num_list, jczx_url, 2, 'jczx')

    # 获取数据
    normal_get_all_resorces(zhk_resource_list, zhk_num_list, zhk_agent)
    print("zhk ok")
    normal_get_all_resorces(jxyx_resource_list, jxyx_num_list, jxyx_agent)
    print("jxyx ok")
    normal_get_all_resorces(jyjg_resource_list, jyjg_num_list, jyjg_agent)
    print("jyjg ok")
    normal_get_all_resorces(jhk_resource_list, jhk_num_list, jhk_agent)
    print("jhk ok")
    normal_get_all_resorces(sjk_resource_list, sjk_num_list, sjk_agent)
    print('sjk ok')
    normal_get_all_resorces(zlb_resource_list, zlb_num_list, zlb_agent)
    print('zlb ok')
    normal_get_all_resorces(tpxqglk_resource_list, tpxqglk_num_list, tpxqglk_agent)
    print('tpxqglk ok')

    other_get_all_resorces(jczx_resource_list, jczx_num_list, jczx_agent)
    print('jczx ok')

    all_list = (zhk_resource_list + jxyx_resource_list + jyjg_resource_list +
                jhk_resource_list + sjk_resource_list + zlb_resource_list +
                tpxqglk_resource_list + jczx_resource_list)

    print('creat list ok')

    df = pd.json_normalize(all_list, sep='_')

    df = df.explode(['attachment_url',
                     'attachment_name',
                     'attachment_webfile_id',
                     'attachment_times']).reset_index(drop=True)

    df.to_csv("fzu_edu.csv", index=False)

    print('to csv ok')

    attach_urls_name = df[["attachment_url", "attachment_name"]]

    folder_path = 'D:/The_Day_I_Became_a_God/Shaddock/3.bugs/fzu_all_attachment'

    for url, name in attach_urls_name.itertuples(index=False):
        if pd.isna(url):
            print("have no attachment")
            continue

        filename = name
        file_path = os.path.join(folder_path, str(filename))
        response = requests.get(url, headers=headers)
        with open(file_path, "wb") as file:
            file.write(response.content)

            print(filename, 'downloads ok')

    print("all_ok")