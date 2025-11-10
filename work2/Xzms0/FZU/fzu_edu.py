from bs4 import BeautifulSoup
from pathlib import Path
import requests
import re
import csv


class FZU:
    def __init__(self):
        '''初始化'''
        self.root_url = "https://jwch.fzu.edu.cn/"
        self.root_dir = Path(__file__).absolute().parent
        
        self.count = 0
        self.page = 1
        self.total_page = None
        self.quantity = 2000 #至少需要获取的数据量

        #写入列表头
        self.file = open(self.root_dir/"data.csv",'w',encoding='utf-8',newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['通知人','标题','日期','链接','附件数','附件下载次数'])

    def make_soup(self,url):
        text = requests.get(url)
        text.encoding = 'utf-8'
        return BeautifulSoup(text.text,'lxml')
    
    def get_clicktimes(self,download_tag):
        '''获取附件下载次数'''
        text = download_tag.get_text()
        #print(text)
        newsid = re.search(r"\(([0-9]{2,}),",text).group(1)
        owner = re.search(r",([0-9]{2,}),",text).group(1)
        #print(newsid)
        #print(owner)
        url = self.root_url+"system/resource/code/news/click/clicktimes.jsp"+\
            f"?wbnewsid={newsid}&owner={owner}&type=wbnewsfile&randomid=nattach"
        download_json = requests.get(url).json()
        #print(download_json)
        return download_json["wbshowtimes"]
    
    def get_detail(self,url):
        '''获取页面详细数据'''
        target_tages = self.make_soup(url).find_all('ul',attrs={'style':"list-style-type:none;"})
        target_attachments = "0"
        target_download = "0_"
        #print(target_tages)
        if target_tages:
            target_download = ""
            attachments_tag=target_tages[0].find_all('li')
            target_attachments = len(attachments_tag)
            for tag in attachments_tag:
                download_tag = tag.find('script')
                value = self.get_clicktimes(download_tag)
                target_download += f"{value},"
                #print(download_tag)
        #print(target_attachments)
        return str(target_attachments),target_download[:-1]
    
    def get_list(self,url):
        '''获取通知条目'''

        print(f"Reading page {self.page}...")
        target_tags = self.make_soup(url).find_all('li')
        #print(target_tags)
        for tag in target_tags:
            date_tag = tag.find('span',class_="doclist_time")
            url_tag = tag.find('a')

            if date_tag and url_tag:
                target_notifier = re.search(r"【(.*)】",tag.get_text()).group(1)
                target_title = url_tag.get("title")
                target_date = re.search(r"[0-9].*[0-9]",date_tag.get_text()).group()
                target_url = self.root_url + url_tag.get("href")
                target_attachments,target_download = self.get_detail(target_url)

                data_unit = [target_notifier,target_title,
                             target_date,target_url,
                             target_attachments,target_download]
                
                self.writer.writerow(data_unit)
                self.count+=1
                #print(data_unit)
                
    def main(self):

        #获取总页数
        soup = self.make_soup(self.root_url+"jxtz.htm")
        last_page = soup.find('a',attrs={'href':'jxtz/1.htm'})
        self.total_page = int(last_page.get_text())
        print(f"There are {self.total_page} pages in total.")

        self.get_list(self.root_url+"jxtz.htm")

        while self.count <= self.quantity:
            self.page += 1
            try:
                self.get_list(self.root_url+f"jxtz/{self.total_page-self.page}.htm")
            except:
                print(f"Read page {self.page} error.")

        self.file.close()


if __name__ == "__main__":
    fzu = FZU()
    fzu.main()

    print(f"Compeleted. \n{fzu.page} pages have been read.")
    print(f"Get {fzu.count} pieces of data successfully!")