from lxml import etree
import requests
import csv


class NotificationData:
    def __init__(self, title, url, author, date):
        self.title = title
        self.url = url
        self.author = author
        self.date = date
        self.attachments = []
    def __str__(self):
        return f'title: {self.title}\nurl: {self.url}\nauthor: {self.author}\ndate: {self.date}\n'

    def __repr__(self):
        return self.__str__()

session = requests.session()
session.trust_env = False #绕过神秘小猫咪代理
session.headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.48 Safari/537.36'
}


def fetch_page(index=None):
    noti_datas = []
    url = 'https://jwch.fzu.edu.cn/jxtz.htm' if index is None else f'https://jwch.fzu.edu.cn/jxtz/{index}.htm' #首页不加页码
    page = session.get(url)
    page = etree.HTML(page.content)
    list_node = page.xpath('/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul')[0]
    nodes = list_node.xpath('./li')
    for i in nodes:
        date = i.xpath('./span/text()')[0].strip().replace('\r',' ').replace('\n', '')
        author = i.xpath('./text()[normalize-space()]')[0].strip().replace('\r',' ').replace('\n', '') #[normalize-space()]过滤空格换行
        title = i.xpath('./a/@title')[0]
        url = 'https://jwch.fzu.edu.cn/' + i.xpath('./a/@href')[0]
        noti_data = NotificationData(title,url,author,date)
        noti_datas.append(noti_data)
    return noti_datas


def get_download_count(wbnewsid,owner,type='wbnewsfile',randomid='nattach'):
    api_url = f'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid={wbnewsid}&owner={owner}&type={type}&randomid={randomid}'
    response = session.get(api_url)
    text = response.text
    if response.status_code == 200:
        start = text.find('"wbshowtimes"')
        end = text.find(',',start)
        count = text[start+14:end]


        return count
    return '0'




def get_attachment(noti_data:NotificationData):
    page = session.get(noti_data.url)
    page_text = page.content.decode('utf-8')
    page = etree.HTML(page.content)
    attachments_node = page.xpath('/html/body/div/div[2]/div[2]/form/div/div[1]/div/ul')

    if attachments_node:
        attachments = attachments_node[0].xpath('./li')
        for i in attachments:
            url = 'https://jwch.fzu.edu.cn/' + i.xpath('./a/@href')[0]
            script = i.xpath('./span/script/text()')[0]
            script = script.split(',')
            #获取附件下载量api参数 owner和randomid其实好像是固定的
            download_wbnewsid = script[0].split('(')[1].strip('"')
            download_owner = script[1].strip('"')
            download_type = script[2].strip('"')
            download_randomid = script[3].split(')')[0].strip('"')


            atta_data = {
                'name': i.xpath('./a/text()')[0],
                'url': url,
                'downloads': get_download_count(download_wbnewsid,download_owner,download_type,download_randomid),
            }
            noti_data.attachments.append(atta_data)

def save_csv(noti_datas:list[NotificationData],filename='notifications.csv'):
    header = ['标题','链接','发布人','发布日期','附件名称','附件链接','附件下载量']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for noti in noti_datas:
            if noti.attachments:
                for atta in noti.attachments: #如果有多个附件,每行写一个,统一美观
                    row = [
                        noti.title,
                        noti.url,
                        noti.author,
                        noti.date,
                        atta['name'],
                        atta['url'],
                        atta['downloads']
                    ]
                    writer.writerow(row)
            else:
                row = [
                    noti.title,
                    noti.url,
                    noti.author,
                    noti.date,
                    '',
                    '',
                    ''
                ]
                writer.writerow(row)

def main(amount):
    main_page = session.get('https://jwch.fzu.edu.cn/jxtz.htm')
    main_page = etree.HTML(main_page.content)
    last_index = main_page.xpath(
        '/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[2]/div[1]/div/span[1]/span[9]/a/text()')[0]  # 最后一页页码
    pages = (amount - 1) // 20 + 1 # 向上取整
    noti_datas = []
    for i in range(pages):
        if i == 0:
            noti_datas = fetch_page()
        else:
            noti_datas += fetch_page(int(last_index) - i)
    for i in noti_datas:
        get_attachment(i)

    save_csv(noti_datas)

if __name__ == '__main__':
    main(500)