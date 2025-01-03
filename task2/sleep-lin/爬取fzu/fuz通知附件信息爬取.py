from lxml import etree  
import requests, time, csv
from lxml import html
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

class FzuSpider(object):
    def __init__(self) -> None:
        self.url = 'https://jwch.fzu.edu.cn/jxtz.htm' 
        self.web = "https://jwch.fzu.edu.cn/jxtz/"
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0'}
        self.urls = []
        self.info = []  # List to store the results
        self.lock = Lock()  # 用于锁定共享资源 self.info，避免并发冲突
    
    def get_urls(self):
        """获取要爬取的网址"""
        tree = html.fromstring(self.try_request_url(self.url).content)
        num = tree.xpath('/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[2]/div[1]/div/span[1]/span[9]/a/text()')
        number = int(input(f"共{num[0]}页,请输入想爬取的页数:"))
        self.urls.append(self.url)
        for i in range(1, number):
            self.urls.append(self.web + f'{int(num[0])-i}.htm')
    
    def spider_urls(self, url):
        """获取网页上全部的链接"""
        req = self.try_request_url(url)
        tree = html.fromstring(req.content)
        li = tree.xpath('//ul[@class="list-gl"]/li')
        for each in li:
            href = each.xpath('.//a/@href')
            man = [text.strip() for text in each.xpath('./text()') if text.strip()]
            time = each.xpath('.//span[@class="doclist_time"]/font/text() | .//span[@class="doclist_time"]/text()')
            time = [t.strip() for t in time if t.strip()]
            if href:
                link = "https://jwch.fzu.edu.cn/" + href[0]
            name,link,exist, nums = self.run(link)
            print(name,link,man[0], time[0], exist, nums)
            with self.lock:  # 使用锁来同步访问共享资源 self.info
                self.info.append([name,link,man[0], time[0], exist, nums])
        print(f"网页{url}爬取成功！")
    
    def run(self, url):
        """爬取某个页面的附件信息"""
        page_content = self.try_request_url(url).content
        tree = etree.HTML(page_content)
        try:
            name = tree.xpath('/html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li/a/text()')
            if not name:
                return '无附件','无附件','无附件', 0
            link = tree.xpath('/html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li/a/@href')
            file_id = link[0].split('=')[-1]
            num = self.get_downLoadTimes(file_id)
        except IndexError as e:
            print(f"发生错误：{e}")
            return '有附件', 0
        return name[0],'https://jwch.fzu.edu.cn/'+link[0],'有附件', num
    
    def get_downLoadTimes(self, id):
        target = 'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp'
        param = {
            'wbnewsid': id,
            'owner': '1744984858',
            'type': 'wbnewsfile',
            'randomid': 'nattach'
        }
        response = self.try_request_url(target, param)
        try:
            times_dic = response.json()
            wbshowtimes = times_dic.get('wbshowtimes')
        except ValueError:
            print("响应内容不是有效的 JSON:", response.text)
            return 0
        return wbshowtimes
    
    def try_request_url(self, url, param=None):
        num = 3
        i = 0
        req = None
        while i < num:
            try:
                req = requests.get(url, headers=self.headers, params=param)
                if req.status_code == 200:
                    return req
                else:
                    print(req.status_code, '尝试重新抓取', url)
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f'网页请求失败：{e}，将重新发起请求...')
                time.sleep(2)
            i += 1
        return None
    
    def save_info(self):
        with open('./附件数据分析.csv', 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['附件名','附件下载链接','通知人', '通知时间', '有无附件', '附件下载次数'])
            for i in self.info:
                writer.writerow(i)
    
    def go(self):
        self.get_urls()
        with ThreadPoolExecutor(max_workers=20) as executor:  
            executor.map(self.spider_urls, self.urls)
        self.save_info()

s = FzuSpider()
s.go()
