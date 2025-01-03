import requests
import re
import csv
from lxml import html
from urllib.parse import parse_qs, urlparse

FJ_list = []
FJN_list = []
FJM_list = []
TITLE_list = []
HOLDER_list = []
DATA_list = []

# 用户代理头部
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.3.5211 SLBChan/112"}

# 打开CSV文件
f = open(r"C:\Users\宋志坤\Desktop\新建文件夹\学习\新建文件夹\qwq.csv", 'w', encoding='utf-8')
writer = csv.writer(f)

# 写入列标题
writer.writerow(["Title", "Holder", "Date", "附件", "附件名", "下载次数"])

# 遍历页面
for num in range(25):  # 翻页
    resp = requests.get(url=f"https://jwch.fzu.edu.cn/jxtz/{196-num}.htm", headers=headers)
    resp.encoding = "utf-8"
    ret = resp.text

    # 使用正则表达式提取数据
    link = re.compile(r'href="\.\.(.*?)"')
    holder = re.compile(r'</span>【(.*?)】')
    data = re.compile(r'(\d{4}-\d{2}-\d{2})')
    title = re.compile(r'"_blank" title="(.*?)"')

    Link = link.findall(ret)
    Title = title.findall(ret)
    Holder = holder.findall(ret)
    Data = data.findall(ret)

    # 填充数据到各自的列表
    for b1 in Title:
        if b1 != '':
            TITLE_list.append(b1)
    for b2 in Holder:
        HOLDER_list.append(b2)
    for b3 in Data:
        DATA_list.append(b3)

    # 处理附件相关的内容
    for _ in range(17):
        Link.pop(0)
    #蠢蛋做法 应该用切片
    for i in Link:
        url1 = f"https://jwch.fzu.edu.cn{i}"
        res2 = requests.get(url1, headers=headers)
        res2.encoding = "utf-8"
        resp2 = html.fromstring(res2.content)
        fj = resp2.xpath("//li/a//text()")
        if fj!=[]:
            if fj[0] in ["综合科","教务处简介"]:
                fj=[]
        fj_link = resp2.xpath('//li/a/@href')
        print("进行爬取中ing")
        if fj_link and fj:
            url=fj_link[0]
            parsed_url=urlparse(url)
            query=parse_qs(parsed_url.query)
            try:
                resp3=requests.get(
                "https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp",
                params={
                "wbnewsid": query["wbfileid"][0],
                "owner": query["owner"][0],
                "type": "wbnewsfile",
                "pandomid": "nattach"
                }
                )
                #print(resp3.json())
                FJN_list.append(resp3.json()['wbshowtimes'])
            except:
                pass
        #fj_num = resp2.xpath("//li/span[@id]//text()")
        else:
            FJN_list.append(0)

        if fj_link==[] or fj==[]:
            FJ_list.append(0)
        else:
            FJ_list.append(f"https://jwch.fzu.edu.cn{fj_link[0]}")
        if fj==[] or fj_link==[]:
            FJM_list.append(0)
        else:
            f_t=fj[0]
            FJM_list.append(f_t)
        # if fj_num==[]:
        #     FJN_list.append("None")
        # else:
        #     k=fj_num[0]
        #     FJN_list.append(k)

# 关闭请求
resp.close()
res2.close()
#这里遇到问题 使用writerow始终会出现填入数值缺失的问题 询问chatgpt 给出下列改动
# 确保所有列表的长度相同
max_len = max(len(TITLE_list), len(HOLDER_list), len(DATA_list), len(FJ_list), len(FJM_list), len(FJN_list))

# 填充较短的列表
TITLE_list.extend([""] * (max_len - len(TITLE_list)))
HOLDER_list.extend([""] * (max_len - len(HOLDER_list)))
DATA_list.extend([""] * (max_len - len(DATA_list)))
FJ_list.extend([""] * (max_len - len(FJ_list)))
FJM_list.extend([""] * (max_len - len(FJM_list)))
FJN_list.extend([""] * (max_len - len(FJN_list)))

# 写入数据行
for i in range(max_len):
    writer.writerow([TITLE_list[i], HOLDER_list[i], DATA_list[i], FJ_list[i], FJM_list[i], FJN_list[i]])

f.close()
