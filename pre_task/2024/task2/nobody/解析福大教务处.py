import re
import sys
import requests
from lxml import etree
import csv
####
'''
1.获取总页数
2.爬取每一页的详细信息
3.定位每页信息的数据
'''

# 1.获取总页数
def Get_page_url(url):
    url = 'https://jwch.fzu.edu.cn/jxtz.htm'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
    }
    page_text = requests.get(url=url, headers=headers).content
    tree= etree.HTML(page_text)
    page_number= tree.xpath('/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[2]/div[1]/div/span[1]/span[4]/a/@href')[0]
    # 匹配字符串中的数字
    page_number = re.findall(r'\d+', page_number)[0]
    # 输出匹配结果
    page_number=int(page_number)+1

    return page_number

# 2.爬取每一页的详细信息
def Get_list(n):
    if n==197:
        url = f"https://jwch.fzu.edu.cn/jxtz.htm"
    else:
        url = f"https://jwch.fzu.edu.cn/jxtz/{n}.htm"

    # print(url)
    headers = {
             'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
    }
    page_text = requests.get(url=url, headers=headers).content
    tree = etree.HTML(page_text)
    list_All = tree.xpath('//ul[@class="list-gl"]/li')
    for li in list_All:

        l=[]
        # 时间
        time = li.xpath('./span/text()')[0]
        time=time.replace('\r','').replace('\n','').replace('\t','').strip()
        # 题目
        title = li.xpath('./a/text()')[0]
        # 部门
        Department =li.xpath('./text()')[1].replace('\n','').replace('\t','').strip()
        res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9]")
        re_string = ''
        Department= res.sub(re_string, Department)
        link = li.xpath('./a/@href')[0]
        link = f"https://jwch.fzu.edu.cn/"+link
        l=[time,title,Department,link]
        l = [Department,title,time,link]
        # 爬取附件
        Attachment(link, l)

def Attachment(link,l):

    headers = {
             'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
    }
    page_text = requests.get(url=link, headers=headers).content
    tree = etree.HTML(page_text)

    try:
        list_All = tree.xpath('/html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li')
    except ZeroDivisionError as e:
        print("捕获到异常:", e)

    else:
        print("没有异常发生，结果为")

    if(len(list_All)>=6):
        print(len(list_All))




    for li in list_All:

        # 时间
        name = li.xpath('./a/text()')[0]
        pattern = r"附件\d+："
        # 替换为空字符串
        name = re.sub(pattern, "", name)
        link = li.xpath('./a/@href')[0]
        link = f"https://jwch.fzu.edu.cn/" + link
        # 下载次数：
        num=li.xpath('./span/script/text()')[0]
        wbnewsid = re.search(r'getClickTimes\((\d*?),', num).group(1)
        url = f'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid={wbnewsid}&owner=1744984858&type=wbnewsfile&randomid=nattach'
        num = requests.get(url).json().get('wbshowtimes')
        l.append(name)
        l.append(num)
        l.append(link)

    writer.writerow(l)




def Download(url):
    a=1


if __name__ == '__main__':

    url = 'https://jwch.fzu.edu.cn/jxtz.htm'
    # 引用csv模块。
    csv_file = open('FZU.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    # 用csv.writer()函数创建一个writer对象。
    l1=['附件名', '附件下载次数', '附件链接']
    l=['通知人', '标题', '日期', '详情链接']
    for n in range(0,9):
        for a in range(0,3):

          l.append(l1[a])

    print(l)
    writer.writerow(l)




    # 1.获取总页数
    All=Get_page_url(url)
    # 2.爬取每一页的详细信息
    All
    for n in range(All-2, All-30,-1):
        Get_list(n)

    csv_file.close()
    sys.exit()





'''
url = 'https://jwch.fzu.edu.cn/jxtz.htm'
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
    }
page_text = requests.get(url=url, headers=headers).content
tree = etree.HTML(page_text)
        #    tree= etree.parse('福大教务处xpath.html',etree.HTMLParser())
   #  r = tree.xpath('/html/head/title')# /代表根节点开始定位，一个层级，
    # r = tree.xpath('/html//title') ”//“代表多个层级，可省略中间的
   # r = tree.xpath('//title') # 可以表示从任意位置开始定位
    # r = tree.xpath('//div[@class="box-gl clearfix"]')# 属性定位，可以定位到特定的标签
    # r = tree.xpath('//div[@class="box-gl clearfix"]/p[3]')  # 索引定位，从一开始
    # r = tree.xpath('//div[@class="box-gl clearfix"]/ul[@class="list-gl"]/li[1]/a/text()')[0]定位到教务处第一条通知，，text()取文本
# r = tree.xpath('//div[@class="box-gl clearfix"]/ul[@class="list-gl"]/li[1]/a/@href')#"/@"获取标签的属性值
#list_All = tree.xpath('//ul[@class="list-gl"]/li[1]/text()')[1]

list_All = tree.xpath('//ul[@class="list-gl"]/li')

for li in list_All:
    title=li.xpath('./a/text()')[0]
    print(title)
'''