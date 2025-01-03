from lxml import etree
import requests
import time
import csv
from multiprocessing import Pool, Manager, cpu_count


class Spider520game(object):
    def __init__(self) -> None:
        self.target = 'https://www.gamer520.com/pcgame'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0'}
        self.urls = []

    def get_urls(self):
        self.urls.append(self.target)
        page_text = self.try_request_url(self.target)
        if not page_text:
            print('未获取到任何信息！')
            return
        page_text = page_text.text
        tree = etree.HTML(page_text)
        num = tree.xpath('//a[@class="page-numbers"]/text()')[-1]
        number = int(input(f'共{num}页，输入想爬取的页数：'))
        for i in range(2, number + 1):
            url = self.target + f'/page/{i}'
            self.urls.append(url)

    def save_info(self, info):
        with open('./520game游戏网.csv', 'a', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['游戏名称', '下载链接'])
            for i in info:
                writer.writerow(i)

    def spider_one_url(self, url, shared_info):
        try:
            page_text = self.try_request_url(url).text
            tree = etree.HTML(page_text)
            a_list = tree.xpath('//a[@rel="bookmark"]')
            for each in a_list:
                title = each.xpath('./text()')[0]
                link = each.xpath('./@href')[0]
                shared_info.append([title, link])
                print(title, link)
        except Exception as e:
            print(f"处理 {url} 时出错: {e}")

    def go(self):
        self.get_urls()
        with Manager() as manager:
            shared_info = manager.list()
            with Pool(cpu_count()) as pool:
                pool.starmap(self.spider_one_url, [(url, shared_info) for url in self.urls])
            self.save_info(shared_info)

    def try_request_url(self, url):
        i = 0
        num = 3
        req = None
        while i < num:
            try:
                req = requests.get(url=url, headers=self.headers)
                if req.status_code == 200:
                    return req
                else:
                    print(req.status_code, '尝试重新抓取', url)
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f'网页请求失败：{e}，将重新发起请求...')
                time.sleep(2)  # 等待 2 秒再尝试
            i += 1
        return req


if __name__ == "__main__":
    s = Spider520game()
    s.go()
