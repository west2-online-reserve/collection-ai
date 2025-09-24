import requests,time,csv
from lxml import html

class FzuSpider(object):
    def __init__(self) -> None:
        self.url = 'https://jwch.fzu.edu.cn/jxtz.htm' 
        self.web = "https://jwch.fzu.edu.cn/jxtz/"
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0'}
        self.urls = []
    
    def get_urls(self):
        """获取要爬取的网址"""
        tree = html.fromstring(self.try_request_url(self.url).content)
        num = tree.xpath('/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[2]/div[1]/div/span[1]/span[9]/a/text()')
        number = int(input(f"共{num[0]}页,请输入想爬取的页数:"))
        self.urls.append(self.url)
        for i in range(1,number):
            #jxtz/194.htm
            self.urls.append(self.web+f'{int(num[0])-i}'+'.htm')

    def spider_one_url(self):
        """爬取某个网页"""
        for url in self.urls:
            req = self.try_request_url(url)
            tree = html.fromstring(req.content)
            li = tree.xpath('//ul[@class = "list-gl"]/li')
            
            with open('fzu通知信息.csv', 'a', newline='', encoding='utf-8') as csvfile:
                # 创建 CSV 写入器
                csvwriter = csv.writer(csvfile)
                
                # 仅在文件为空时写入表头
                if csvfile.tell() == 0:
                    csvwriter.writerow(['标题', '通知人', '时间', '链接'])
                
                for each in li:
                    # 获取标题
                    title = each.xpath('.//a/text()')
                    # 获取通知人
                    man = [text.strip() for text in each.xpath('./text()') if text.strip()]
                    # 获取时间，优先从 font 标签中获取
                    time = each.xpath('.//span[@class="doclist_time"]/font/text() | .//span[@class="doclist_time"]/text()')
                    time = [t.strip() for t in time if t.strip()]  # 清理空格
                    # 获取链接
                    href = each.xpath('.//a/@href')
                    
                    # 准备行数据
                    row = [
                        title[0] if title else "无标题信息",
                        man[0] if man else "无通知人信息",
                        time[0] if time else "无时间信息",
                        "https://jwch.fzu.edu.cn/" + href[0] if href else "无链接信息"
                    ]
                    
                    # 写入行
                    csvwriter.writerow(row)  
            print(f"网页{url} 爬取成功！")
    

    def go(self):
        self.get_urls()
        self.spider_one_url()

    def try_request_url(self, url):
        num = 3
        i = 0
        req = None
        while i < num:
            try:
                req = requests.get(url, headers=self.headers)
                if req.status_code == 200:
                    return req  # 请求成功，返回结果
                else:
                    print(req.status_code, '尝试重新抓取', url)
                    time.sleep(2)  # 等待 2 秒再尝试
            except requests.exceptions.RequestException as e:
                print(f'网页请求失败：{e}，将重新发起请求...')
                time.sleep(2)  # 等待 2 秒再尝试
            
            i += 1  # 只在请求失败时增加计数器

        return None  # 如果所有尝试都失败，返回 None

s = FzuSpider()
s.go()