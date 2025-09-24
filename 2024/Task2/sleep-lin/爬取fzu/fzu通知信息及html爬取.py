import requests,os,time,csv
from multiprocessing import Pool,cpu_count
from bs4 import BeautifulSoup
from lxml import html

class Spiderfzu(object):
    def __init__(self) -> None:
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0'}
        self.target =  'https://jwch.fzu.edu.cn/jxtz.htm'
        self.web = 'https://jwch.fzu.edu.cn/'
        self.urls = []#爬取的页面网址
        self.notice_urls = []#页面上通知的网址
        self.save_dir = './fzu通知'

    def try_request_url(self,url):
        num = 3
        i = 0
        req = None
        while i<num:
            try:
                req = requests.get(url,headers=self.headers)
                if req.status_code == 200:
                    break
                else:
                    print(req.status_code,'尝试重新抓取',url)
                    print(req.text)
                    time.sleep(2)
                    i+=1
            except:
                print('网页请求失败！将重新发起请求...')
        return req

    def get_page_urls(self):
        """存储每页的网址"""
        req = self.try_request_url(self.target)
        b = BeautifulSoup(req.content,'lxml')
        page_num = int(b.find_all('span',class_='p_no')[-1].string)
        number = int(input(f"共{page_num}页,请输入想爬取的页数:"))
        self.urls.append(self.target)
        for i in range(1,number):
            #jxtz/194.htm
            self.urls.append(self.web+'jxtz/'+f'{page_num-i}'+'.htm')
    
    def get_notice_urls(self,url):
        """存储通知的网址"""
        req = self.try_request_url(url)
        b = BeautifulSoup(req.content,'lxml')
        ul_tags = b.find_all('ul',class_="list-gl")
        for ul in ul_tags:
            li_tags = ul.find_all('li')
            for li in li_tags:
                # 找到 span 和 a 标签之间的文本
                span_tag = li.find('span', class_='doclist_time')
                a_tag = li.find('a')
                if span_tag and a_tag:
                    text_between = span_tag.find_next_sibling(text=True).strip()
                    print("时间:", span_tag.text.strip())
                    print("通知人:", text_between)
                    print("链接文本:", a_tag.text.strip())
                    print("链接:", 'https://jwch.fzu.edu.cn/'+a_tag['href'])
                    self.notice_urls.append('https://jwch.fzu.edu.cn/'+a_tag['href'])
                    print('-' * 30)

    def spider_one_url(self,url):
        """爬取具体通知内容"""
        req = self.try_request_url(url)
        b = BeautifulSoup(req.content,'lxml')
        title = b.find('title').string if b.find('title') else '无标题'
        if not os.path.exists('./fzu通知'):
            os.mkdir('./fzu通知')
        try:
            with open(f"./fzu通知/{title}.txt", 'w', encoding='utf-8') as pf:
                pf.write(f"标题: {title}\n\n")
                div_tags = b.find('div',id="vsb_content")
                #print(f"进程{os.getpid()},处理{title}") 可用于查看是否为多线程处理
                if div_tags:
                    p_tags = div_tags.find_all('p')
                    print(title,'保存成功！')
                    for each in p_tags:
                        text = each.get_text(strip=True)
                        if text:  # 确保内容不为空
                            pf.write(text + "\n")
                else:
                    print('未找到内容！')
        except:
            pass
        
    def save_info(self):
        """储存通知的标题、通知人、时间、链接"""
        for url in self.urls:
            req = self.try_request_url(url)
            tree = html.fromstring(req.content)
            li = tree.xpath('//ul[@class = "list-gl"]/li')
            
            if not os.path.exists('./fzu通知'):
                os.mkdir('./fzu通知')
            with open('./fzu通知/通知信息汇总.csv', 'a', newline='', encoding='utf-8') as csvfile:
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
            print(f"网页{url}爬取成功！")

    def go(self):
        self.get_page_urls()
        self.save_info()
        for each in self.urls:
           self.get_notice_urls(each)
        print(f"获取到的通知 URL 数量: {len(self.notice_urls)}")
        with Pool(cpu_count()) as p:
            p.map(self.spider_one_url,self.notice_urls)


if __name__ == "__main__":
    spider = Spiderfzu()
    spider.go()


        

                
